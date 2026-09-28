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
        # Encode source code
        if self.encoder and self.tokenizer:
            encoded = self.tokenizer(
                source_code,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            )
            encoder_output = self.encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )[0]  # [batch_size, seq_len, hidden_size]
        else:
            # Fallback: simple embedding
            encoder_output = gnn_embeddings.unsqueeze(1)
        
        # Project GNN embeddings to the transformer's hidden size if needed
        if gnn_embeddings.size(-1) != self.hidden_size:
            gnn_embeddings = self.gnn_projection(gnn_embeddings)

        # Fuse GNN embeddings with encoder output via attention
        gnn_expanded = gnn_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        fused_output, attention_weights = self.fusion_layer(
            query=gnn_expanded,
            key=encoder_output,
            value=encoder_output
        )
        
        # Generate refactored code
        generated_tokens = self.decoder(
            encoder_output=encoder_output,
            gnn_context=fused_output,
            refactoring_type=refactoring_type
        )
        
        return generated_tokens, attention_weights
    
    def generate(self, 
                 source_code: str,
                 gnn_embeddings: torch.Tensor,
                 refactoring_type: int = 0,
                 max_length: int = 512,
                 temperature: float = 1.0) -> str:
        """
        Generate refactored code (greedy decoding)
        
        Args:
            source_code: Input source code
            gnn_embeddings: Structural embeddings
            refactoring_type: Type of refactoring
            max_length: Maximum generation length
            temperature: Sampling temperature
        
        Returns:
            Generated refactored code
        """
        self.eval()
        
        with torch.no_grad():
            # Project GNN embeddings if needed for fallback path
            if gnn_embeddings.size(-1) != self.hidden_size:
                gnn_embeddings = self.gnn_projection(gnn_embeddings)

            # Encode
            if self.encoder and self.tokenizer:
                encoded = self.tokenizer(
                    [source_code],
                    return_tensors="pt",
                    max_length=self.max_seq_length,
                    truncation=True
                )
                encoder_output = self.encoder(**encoded)[0]
            else:
                if gnn_embeddings.size(-1) != self.hidden_size:
                    gnn_embeddings = self.gnn_projection(gnn_embeddings)
                encoder_output = gnn_embeddings.unsqueeze(1)
            
            # Decode greedily
            generated_ids = []
            current_token = torch.tensor([[2]])  # BOS token
            
            for _ in range(max_length):
                # Simple generation (can be enhanced with beam search)
                logits = self.decoder.predict_next_token(
                    encoder_output=encoder_output,
                    gnn_context=gnn_embeddings.unsqueeze(1),
                    current_tokens=current_token,
                    refactoring_type=refactoring_type
                )
                
                next_token = torch.argmax(logits, dim=-1)
                generated_ids.append(next_token.item())
                
                if next_token.item() == 3:  # EOS token
                    break
                
                current_token = next_token
        
        # Decode to text
        if self.tokenizer:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
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
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = self._create_positional_encoding(512, hidden_size)
        
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
            # Teacher forcing with special tokens
            tgt_embed = self.embedding(torch.ones((encoder_output.size(0), 1), dtype=torch.long) * 2)
        
        # Add positional encoding
        tgt_embed = tgt_embed + self.positional_encoding[:tgt_embed.size(1), :].unsqueeze(0)
        
        # Decode with GNN context
        memory = torch.cat([encoder_output, gnn_context], dim=1)
        decoded = self.decoder_stack(tgt_embed, memory)
        
        # Project to vocabulary
        logits = self.output_projection(decoded)
        
        return logits
    
    def predict_next_token(self,
                          encoder_output: torch.Tensor,
                          gnn_context: torch.Tensor,
                          current_tokens: torch.Tensor,
                          refactoring_type: int = 0) -> torch.Tensor:
        """Predict next token during generation"""
        tgt_embed = self.embedding(current_tokens)
        tgt_embed = tgt_embed + self.positional_encoding[:tgt_embed.size(1), :].unsqueeze(0)
        
        memory = torch.cat([encoder_output, gnn_context], dim=1)
        decoded = self.decoder_stack(tgt_embed, memory)
        
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
        
        Returns:
            Dictionary with refactored code, type, and confidence
        """
        generated_code = self.model.generate(
            source_code=source_code,
            gnn_embeddings=gnn_embeddings,
            refactoring_type=refactoring_type,
            max_length=max_length
        )
        
        return {
            "original_code": source_code,
            "refactored_code": generated_code,
            "refactoring_type": self.refactoring_types.get(refactoring_type, "unknown"),
            "confidence": float(confidence_score),
            "type_id": refactoring_type
        }
