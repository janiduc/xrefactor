"""
Fine-tunes CodeTransformer's decoder (+ fusion layer + GNN projection) on the
before/after code pairs mined by refactoring_mining/build_codegen_dataset.py,
via standard teacher-forced, causally-masked next-token cross-entropy.

GNN-context placeholder (a documented, visible simplification, not a hidden
one): real per-example GNN structural embeddings for the exact historical
code element would require rebuilding a CPG at every mined commit (the same
expensive per-commit rebuild build_refactoring_dataset.py already does for
its ~3700 embedding-labeled examples) - doing that again for all ~3000
codegen pairs was judged too slow for a first training pass. Instead, this
script learns a small `type_embedding: nn.Embedding(10, gnn_embedding_dim)`
that stands in for "gnn_embeddings" - i.e. the model is conditioned on
WHICH refactoring pattern to apply, not on the specific structural
neighborhood of the input method. This flows through CodeTransformer's
existing forward()/generate() exactly as real GNN embeddings would (same
shapes, same fusion_layer/gnn_projection code path), so upgrading to real
per-example embeddings later is a drop-in replacement, not a redesign.
Expect generation quality to reflect this: the model learns per-pattern
transformation style more than per-instance structural context.

The pretrained CodeBERT encoder is FROZEN (only the decoder, fusion layer,
gnn_projection, and type_embedding train) - fine-tuning a 125M-parameter
encoder on ~3000 examples would be far more likely to overfit/destabilize
than to help, and freezing keeps each step's backward pass cheap enough to
run on CPU.

Training is per-example (batch_size=1): source/target sequences vary
widely in length, and implementing correct padding/attention masks for
batched training through both the frozen CodeBERT encoder and the causal
decoder was judged not worth the complexity for a first training pass at
this data scale. A natural follow-up if training needs to scale further.

Usage:
    python -m src.transformer.train_transformer \
        --dataset ./refactoring_mining/codegen_pairs.jsonl \
        --output ./models/transformer_trained.pt \
        --epochs 3 --device cpu
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.gnn.refactoring_types import REFACTORING_TYPE_TO_ID
from src.transformer.code_generator import CodeTransformer
from src.utils.data_split import split_indices
from src.utils.helpers import ConfigManager
from src.utils.metrics import aggregate_generation_quality


def _load_pairs(dataset_path: Path) -> List[dict]:
    pairs = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _forward_loss(model: CodeTransformer, type_embedding: nn.Embedding,
                   before: str, after: str, refactoring_type_id: int,
                   max_seq_length: int, device: str) -> torch.Tensor:
    encoded_src = model.tokenizer([before], return_tensors="pt", truncation=True,
                                   max_length=max_seq_length).to(device)
    encoder_output = model.encoder(**encoded_src)[0]

    gnn_emb = type_embedding(torch.tensor([refactoring_type_id], device=device))
    # Shared with generation via CodeTransformer.build_gnn_context, so the
    # decoder's memory slot has the same distribution at train and inference
    # time. Previously this path fused while generate() passed the raw
    # projected embedding, leaving fusion_layer's trained weights unused.
    fused_output, _ = model.build_gnn_context(encoder_output, gnn_emb)

    target_ids = model.tokenizer([after], return_tensors="pt", truncation=True,
                                  max_length=max_seq_length)["input_ids"].to(device)
    if target_ids.size(1) < 2:
        return None  # nothing to predict (empty/degenerate target)

    decoder_input = target_ids[:, :-1]
    decoder_target = target_ids[:, 1:]

    logits = model.decoder(encoder_output=encoder_output, gnn_context=fused_output, target_tokens=decoder_input)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))


def train(dataset_path: Path, output_path: Path, config_path: str,
          epochs: int = 3, lr: float = 1e-4, device: str = "cpu",
          model_name: str = "microsoft/codebert-base", max_seq_length: int = 128,
          max_train_examples: int = None, eval_sample_size: int = 60, eval_max_length: int = 80):
    pairs = _load_pairs(dataset_path)
    if len(pairs) < 10:
        raise ValueError(f"Only {len(pairs)} pairs - build more before training")

    config = ConfigManager(config_path)
    train_idx, val_idx, test_idx = split_indices(len(pairs), config)
    train_idx, val_idx, test_idx = train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
    if max_train_examples:
        # Bounds wall-clock time (no batching + a frozen-but-still-forward-passed
        # 125M-param encoder makes each step CPU-costly); val/test stay at their
        # natural split size so evaluation is still meaningful.
        train_idx = train_idx[:max_train_examples]
    logger.info(f"Training on {len(train_idx)} pairs, validating on {len(val_idx)}, "
                f"held-out test {len(test_idx)}")

    model = CodeTransformer(model_name=model_name, max_seq_length=max_seq_length).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()

    type_embedding = nn.Embedding(len(REFACTORING_TYPE_TO_ID), model.gnn_embedding_dim).to(device)

    trainable_params = (
        list(model.decoder.parameters()) + list(model.fusion_layer.parameters())
        + list(model.gnn_projection.parameters()) + list(type_embedding.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    id_to_type = {v: k for k, v in REFACTORING_TYPE_TO_ID.items()}

    for epoch in range(1, epochs + 1):
        model.decoder.train()
        model.fusion_layer.train()
        type_embedding.train()

        total_loss, num_steps = 0.0, 0
        for i in train_idx:
            pair = pairs[i]
            loss = _forward_loss(model, type_embedding, pair["before"], pair["after"],
                                  pair["refactoring_type_id"], max_seq_length, device)
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

        avg_train_loss = total_loss / max(1, num_steps)

        model.decoder.eval()
        model.fusion_layer.eval()
        type_embedding.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for i in val_idx:
                pair = pairs[i]
                loss = _forward_loss(model, type_embedding, pair["before"], pair["after"],
                                      pair["refactoring_type_id"], max_seq_length, device)
                if loss is not None:
                    val_loss += loss.item()
                    val_steps += 1
        avg_val_loss = val_loss / max(1, val_steps)

        logger.info(f"epoch {epoch}/{epochs} - train loss: {avg_train_loss:.4f} - val loss: {avg_val_loss:.4f}")

    # Qualitative + quantitative check on the held-out test split: generate greedily
    # and compare to the real "after" text with Phase 0's generation_quality metrics.
    # Greedy generation is autoregressive (one full decoder forward pass PER output
    # token, no KV-caching) and an early-training decoder rarely predicts EOS, so it
    # tends to run all the way to eval_max_length every time - bounding both the
    # per-generation length AND how many test examples get generated is necessary
    # to keep this tractable on CPU; this is a deliberately bounded check, not an
    # exhaustive sweep of the whole held-out test split.
    eval_indices = test_idx[:eval_sample_size]
    model.eval()
    generation_samples = []
    per_type_pairs: Dict[str, List[dict]] = {}
    for i in eval_indices:
        pair = pairs[i]
        gnn_emb = type_embedding(torch.tensor([pair["refactoring_type_id"]], device=device))
        with torch.no_grad():
            generated = model.generate(
                source_code=pair["before"], gnn_embeddings=gnn_emb,
                refactoring_type=pair["refactoring_type_id"], max_length=eval_max_length,
            )
        type_name = id_to_type[pair["refactoring_type_id"]]
        per_type_pairs.setdefault(type_name, []).append({"reference": pair["after"], "hypothesis": generated})
        if len(generation_samples) < 10:
            generation_samples.append({
                "refactoring_type": type_name, "before": pair["before"][:300],
                "expected_after": pair["after"][:300], "generated": generated[:300],
            })

    per_type_quality = {t: aggregate_generation_quality(ps) for t, ps in per_type_pairs.items()}
    overall_quality = aggregate_generation_quality([p for ps in per_type_pairs.values() for p in ps])
    logger.info(f"Held-out test generation quality (overall): {overall_quality}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save only the submodules that actually trained - the frozen 125M-param
    # CodeBERT encoder is always freshly loaded from HuggingFace by
    # CodeTransformer's constructor, so persisting it here would bloat the
    # checkpoint by ~500MB for no benefit.
    torch.save({
        "decoder_state": model.decoder.state_dict(),
        "fusion_layer_state": model.fusion_layer.state_dict(),
        "gnn_projection_state": model.gnn_projection.state_dict(),
        "type_embedding_state": type_embedding.state_dict(),
        "tokenizer_name": model_name,
        "vocab_size": model.vocab_size,
        "gnn_embedding_dim": model.gnn_embedding_dim,
        "config": {"hidden_size": model.hidden_size, "num_layers": model.num_layers,
                   "max_seq_length": max_seq_length},
    }, output_path)
    logger.info(f"Saved trained transformer to {output_path}")

    metrics_path = output_path.parent / (output_path.stem + ".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "dataset": str(dataset_path), "num_pairs": len(pairs),
            "train_size": len(train_idx), "val_size": len(val_idx), "test_size": len(test_idx),
            "eval_sample_size": len(eval_indices), "eval_max_length": eval_max_length,
            "overall_test_generation_quality": overall_quality,
            "per_type_test_generation_quality": per_type_quality,
            "sample_generations": generation_samples,
        }, f, indent=2)
    logger.info(f"Saved generation-quality report to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeTransformer's decoder on mined before/after pairs")
    parser.add_argument("--dataset", default="./refactoring_mining/codegen_pairs.jsonl")
    parser.add_argument("--output", default="./models/transformer_trained.pt")
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--max-train-examples", type=int, default=None,
                         help="Cap the number of TRAIN examples per epoch (val/test keep their natural split size). "
                              "Useful to bound wall-clock time on CPU; omit to use the full train split.")
    parser.add_argument("--eval-sample-size", type=int, default=30,
                         help="How many held-out test examples to run full greedy generation on (bounded - "
                              "generation is autoregressive with no KV-caching, ~0.3s/token on CPU).")
    parser.add_argument("--eval-max-length", type=int, default=40,
                         help="Max tokens to generate per evaluation example.")
    args = parser.parse_args()

    train(Path(args.dataset), Path(args.output), args.config, epochs=args.epochs, lr=args.lr,
          device=args.device, model_name=args.model_name, max_seq_length=args.max_seq_length,
          max_train_examples=args.max_train_examples, eval_sample_size=args.eval_sample_size,
          eval_max_length=args.eval_max_length)


if __name__ == "__main__":
    main()
