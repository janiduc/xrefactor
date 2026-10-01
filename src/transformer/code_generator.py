"""
Stage 3: Transformer-based Code Generation
Generates refactored code via Transformer conditioned on GNN's structural understanding.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from typing import Dict, List, Tuple, Optional, Any
import json


def _apply_repetition_penalty(logits: torch.Tensor,
                              generated_ids: List[int],
                              penalty: float) -> torch.Tensor:
    """Divide the logits of already-emitted tokens by `penalty` (>1 discourages
    repetition). Negative logits are multiplied instead, so the penalty always
    pushes a token DOWN regardless of sign."""
    if penalty is None or penalty == 1.0 or not generated_ids:
        return logits
    logits = logits.clone()
    for token_id in set(generated_ids):
        value = logits[0, token_id]
        logits[0, token_id] = value / penalty if value > 0 else value * penalty
    return logits


def _block_repeated_ngrams(logits: torch.Tensor,
                           generated_ids: List[int],
                           ngram_size: int) -> torch.Tensor:
    """Forbid any continuation that would repeat an already-seen n-gram.

    An undertrained decoder's dominant failure mode is a short cycle; blocking
    repeated n-grams breaks the cycle rather than masking it.
    """
    if not ngram_size or ngram_size < 2 or len(generated_ids) < ngram_size:
        return logits
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned = {
        generated_ids[i + ngram_size - 1]
        for i in range(len(generated_ids) - ngram_size + 1)
        if tuple(generated_ids[i:i + ngram_size - 1]) == prefix
    }
    if not banned:
        return logits
    logits = logits.clone()
    for token_id in banned:
        logits[0, token_id] = float("-inf")
    return logits


class CodeTransformer(nn.Module):
    """
    Transformer-based code generator for refactoring
    Combines structural understanding from GNN with code generation
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/codebert-base",
                 gnn_embedding_dim: int = 128,
                 vocab_size: int = 50000,
                 hidden_size: int = 768,
                 num_layers: int = 6,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 dropout: float = 0.1,
                 max_seq_length: int = 512):
        """
        Args:
            model_name: Pre-trained model name (CodeBERT, GraphCodeBERT, etc.)
            vocab_size: Vocabulary size for token embeddings
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: FFN intermediate size
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(CodeTransformer, self).__init__()
        
        self.model_name = model_name
        self.gnn_embedding_dim = gnn_embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        logger.info(f"Loading pre-trained model: {model_name}")

        # Load pre-trained encoder
        try:
            self.encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}. Using default implementation.")
            self.encoder = None
            self.tokenizer = None

        # The decoder's vocabulary MUST match the tokenizer's real vocabulary - a
        # constructor default (50000) that doesn't match CodeBERT's actual
        # tokenizer (50265, with different special-token ids) means predicted
        # token ids get decoded against the wrong vocabulary, producing
        # incoherent subword garbage regardless of training. Prefer the real
        # tokenizer's size whenever one loaded successfully.
        if self.tokenizer is not None:
            vocab_size = len(self.tokenizer)
        self.vocab_size = vocab_size

        # Decoder for generation
        self.decoder = TransformerDecoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            vocab_size=vocab_size
        )
        
        # Project GNN embeddings to transformer hidden size if needed
        if self.gnn_embedding_dim != self.hidden_size:
            self.gnn_projection = nn.Linear(self.gnn_embedding_dim, self.hidden_size)
        else:
            self.gnn_projection = nn.Identity()
        
        # Attention fusion layer (combines encoder and GNN outputs)
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        logger.info("CodeTransformer initialized successfully")
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_source(self, source_code: List[str], gnn_embeddings: torch.Tensor) -> torch.Tensor:
        """Run the (frozen) pretrained encoder over source text.

        Falls back to treating the GNN embedding as a length-1 memory when no
        pretrained encoder/tokenizer could be loaded.
        """
        if self.encoder and self.tokenizer:
            encoded = self.tokenizer(
                source_code,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            return self.encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )[0]  # [batch_size, seq_len, hidden_size]
        return gnn_embeddings.unsqueeze(1) if gnn_embeddings.dim() == 2 else gnn_embeddings

    def build_gnn_context(self,
                          encoder_output: torch.Tensor,
                          gnn_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project the GNN embedding and fuse it with the encoder output.

        SINGLE SOURCE OF TRUTH for the decoder's `gnn_context` memory slot,
        shared by training (src/transformer/train_transformer.py) and
        generation. Previously training passed this fused output while
        generation passed the raw projected embedding, so `fusion_layer`'s
        trained weights were never used at inference and the decoder saw an
        out-of-distribution memory - one of the reasons generated code was
        meaningless.
        """
        if gnn_embeddings.size(-1) != self.hidden_size:
            gnn_embeddings = self.gnn_projection(gnn_embeddings)
        gnn_expanded = gnn_embeddings.unsqueeze(1) if gnn_embeddings.dim() == 2 else gnn_embeddings
        fused_output, attention_weights = self.fusion_layer(
            query=gnn_expanded,
            key=encoder_output,
            value=encoder_output
        )
        return fused_output, attention_weights

    def forward(self,
                source_code: List[str],
                gnn_embeddings: torch.Tensor,
                refactoring_type: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate refactored code

        Args:
            source_code: List of source code snippets
            gnn_embeddings: Structural embeddings from GNN [batch_size, hidden_size]
            refactoring_type: Type of refactoring to apply [batch_size]

        Returns:
            generated_tokens: Generated code tokens [batch_size, seq_len]
            attention_weights: Attention weights for interpretability
        """
        encoder_output = self.encode_source(source_code, gnn_embeddings)
        fused_output, attention_weights = self.build_gnn_context(encoder_output, gnn_embeddings)

        generated_tokens = self.decoder(
            encoder_output=encoder_output,
            gnn_context=fused_output,
            refactoring_type=refactoring_type
        )

        return generated_tokens, attention_weights
    
    def special_token_ids(self) -> Tuple[int, int]:
        """(bos, eos) from the real tokenizer when available.

        Hardcoded ids would not match CodeBERT's actual special-token layout -
        its vocabulary differs from the generic constants 2/3 this once assumed.
        """
        bos_id, eos_id = 2, 3
        if self.tokenizer is not None:
            if self.tokenizer.bos_token_id is not None:
                bos_id = self.tokenizer.bos_token_id
            if self.tokenizer.eos_token_id is not None:
                eos_id = self.tokenizer.eos_token_id
        return bos_id, eos_id

    def generate(self,
                 source_code: str,
                 gnn_embeddings: torch.Tensor,
                 refactoring_type: int = 0,
                 max_length: int = 512,
                 temperature: float = 1.0,
                 repetition_penalty: float = 1.2,
                 no_repeat_ngram_size: int = 3) -> str:
        """
        Generate refactored code (greedy decoding with anti-repetition).

        The decoding loop feeds the decoder the WHOLE prefix generated so far.
        It previously assigned `current_token = next_token`, which threw the
        prefix away (leaving the decoder with no memory of what it had already
        emitted, so it could not avoid repeating itself) and collapsed the
        tensor to 1-D, making `positional_encoding[:emb.size(1)]` slice by
        hidden_size instead of sequence length and feeding the decoder a
        meaningless 512-step broadcast. That is why output looked like
        "private final private final private final ...".

        Args:
            source_code: Input source code
            gnn_embeddings: Structural embeddings
            refactoring_type: Type of refactoring
            max_length: Maximum number of tokens to generate
            temperature: Logit temperature (1.0 = plain greedy)
            repetition_penalty: >1.0 divides the logits of already-emitted
                tokens, discouraging loops
            no_repeat_ngram_size: forbids repeating any n-gram of this size

        Returns:
            Generated refactored code
        """
        self.eval()

        with torch.no_grad():
            encoder_output = self.encode_source([source_code], gnn_embeddings)
            # Same fused context the training path builds - see build_gnn_context.
            gnn_context, _ = self.build_gnn_context(encoder_output, gnn_embeddings)

            bos_id, eos_id = self.special_token_ids()
            position_limit = min(self.decoder.max_positions, self.max_seq_length)

            generated_ids: List[int] = []
            current_tokens = torch.tensor([[bos_id]], dtype=torch.long, device=self.device)

            for _ in range(max_length):
                logits = self.decoder.predict_next_token(
                    encoder_output=encoder_output,
                    gnn_context=gnn_context,
                    current_tokens=current_tokens,
                    refactoring_type=refactoring_type
                )  # [1, vocab_size]

                logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)
                logits = _block_repeated_ngrams(logits, generated_ids, no_repeat_ngram_size)
                if temperature and temperature != 1.0:
                    logits = logits / temperature

                next_token = torch.argmax(logits, dim=-1)  # [1]
                token_id = int(next_token.item())
                if token_id == eos_id:
                    break

                generated_ids.append(token_id)
                # Keep the prefix 2-D ([batch, seq_len]) and GROWING.
                current_tokens = torch.cat([current_tokens, next_token.view(1, 1)], dim=1)
                if current_tokens.size(1) >= position_limit:
                    break

        if self.tokenizer:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return " ".join(str(tid) for tid in generated_ids)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for generating refactored code
    """
    
    def __init__(self,
                 hidden_size: int = 768,
                 num_layers: int = 6,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 dropout: float = 0.1,
                 vocab_size: int = 50000):
        super(TransformerDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_positions = 512  # rows in the positional encoding; generation must not exceed it

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = self._create_positional_encoding(self.max_positions, hidden_size)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_stack = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding"""
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, 
                encoder_output: torch.Tensor,
                gnn_context: torch.Tensor,
                refactoring_type: Optional[List[int]] = None,
                target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for decoder
        
        Args:
            encoder_output: Encoder output [batch_size, src_len, hidden_size]
            gnn_context: GNN context [batch_size, 1, hidden_size]
            refactoring_type: Refactoring type indicators
            target_tokens: Target tokens during training
        
        Returns:
            logits: Output logits [batch_size, tgt_len, vocab_size]
        """
        # Embed target tokens
        if target_tokens is not None:
            tgt_embed = self.embedding(target_tokens)
        else:
            # No target supplied: start from a single BOS-like position.
            start = torch.ones((encoder_output.size(0), 1), dtype=torch.long,
                               device=self.embedding.weight.device) * 2
            tgt_embed = self.embedding(start)

        # Add positional encoding. `positional_encoding` is a plain tensor rather
        # than a registered buffer (registering it would add a state_dict key and
        # break already-saved checkpoints), so move it per use instead.
        positional = self.positional_encoding[:tgt_embed.size(1), :].unsqueeze(0).to(tgt_embed.device)
        tgt_embed = tgt_embed + positional

        # Causal mask: without this, nn.TransformerDecoder's self-attention over
        # tgt_embed can attend to LATER positions too, so teacher forcing would
        # trivially "cheat" by looking at the very token it's supposed to
        # predict - the loss would drop sharply but the model would never learn
        # genuine autoregressive generation (at real inference time, future
        # tokens don't exist yet).
        tgt_len = tgt_embed.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_embed.device)

        # Decode with GNN context
        memory = torch.cat([encoder_output, gnn_context], dim=1)
        decoded = self.decoder_stack(tgt_embed, memory, tgt_mask=causal_mask)

        # Project to vocabulary
        logits = self.output_projection(decoded)

        return logits
    
    def predict_next_token(self,
                          encoder_output: torch.Tensor,
                          gnn_context: torch.Tensor,
                          current_tokens: torch.Tensor,
                          refactoring_type: int = 0) -> torch.Tensor:
        """Predict the next token given the WHOLE prefix generated so far.

        `current_tokens` must be [batch_size, seq_len]. The assert is deliberate:
        passing a 1-D tensor silently "works" via broadcasting but makes
        `positional_encoding[:tgt_embed.size(1)]` slice by hidden_size rather
        than sequence length, feeding the stack a meaningless 512-step tensor.
        That was a real bug; fail loudly instead.
        """
        assert current_tokens.dim() == 2, (
            f"current_tokens must be [batch, seq_len], got shape {tuple(current_tokens.shape)}"
        )
        seq_len = current_tokens.size(1)
        if seq_len > self.max_positions:
            raise ValueError(f"prefix length {seq_len} exceeds max_positions {self.max_positions}")

        tgt_embed = self.embedding(current_tokens)
        positional = self.positional_encoding[:seq_len, :].unsqueeze(0).to(tgt_embed.device)
        tgt_embed = tgt_embed + positional

        # Causal mask, matching forward(): without it the prefix attends
        # bidirectionally here but causally during training, so the two paths
        # would see different distributions.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_embed.device)

        memory = torch.cat([encoder_output, gnn_context], dim=1)
        decoded = self.decoder_stack(tgt_embed, memory, tgt_mask=causal_mask)

        logits = self.output_projection(decoded[:, -1:, :])

        return logits.squeeze(1)


class RefactoringGenerator:
    """
    High-level interface for generating refactoring suggestions
    """
    
    def __init__(self, transformer_model: CodeTransformer, device: str = "cuda"):
        self.model = transformer_model.to(device)
        self.device = device
        self.refactoring_types = {
            0: "extract_method",
            1: "move_class",
            2: "rename_variable",
            3: "consolidate_duplicate_code",
            4: "remove_dead_code",
            5: "simplify_condition",
            6: "split_class",
            7: "extract_interface",
            8: "reduce_coupling",
            9: "improve_naming"
        }
    
    def suggest_refactoring(self,
                           source_code: str,
                           gnn_embeddings: torch.Tensor,
                           confidence_score: float,
                           refactoring_type: int = 0,
                           max_length: int = 512) -> Dict[str, Any]:
        """
        Generate refactoring suggestion

        Runs the generated code through src/transformer/pattern_validators.py's
        structural check for this pattern before returning - a suggestion whose
        generated code doesn't structurally look like a correct instance of the
        pattern gets its confidence discounted and an explicit caveat attached,
        rather than being presented as if it were reliable.

        Returns:
            Dictionary with refactored code, type, confidence, and
            structural_validation (see pattern_validators.validate).
        """
        generated_code = self.model.generate(
            source_code=source_code,
            gnn_embeddings=gnn_embeddings,
            refactoring_type=refactoring_type,
            max_length=max_length
        )

        pattern_name = self.refactoring_types.get(refactoring_type, "unknown")
        try:
            from src.transformer.pattern_validators import validate
            validation = validate(pattern_name, source_code, generated_code)
        except Exception as e:
            validation = {"passed": False, "reason": f"validation error: {e}"}

        effective_confidence = float(confidence_score)
        if not validation.get("passed"):
            effective_confidence *= 0.3  # discount, don't hide - the raw score is still visible below

        return {
            "original_code": source_code,
            "refactored_code": generated_code,
            "refactoring_type": pattern_name,
            "confidence": effective_confidence,
            "raw_confidence": float(confidence_score),
            "type_id": refactoring_type,
            "structural_validation": validation,
        }
