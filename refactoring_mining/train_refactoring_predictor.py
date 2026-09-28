"""
Supervised fine-tuning of RefactoringPredictor's classifier + confidence heads
using the labeled (embedding, refactoring_type) dataset produced by
build_refactoring_dataset.py.

The classifier is trained with standard cross-entropy against the mined
labels. The confidence head is trained to predict whether the classifier's
own top prediction was correct (a common calibration trick), rather than
being hand-set to an arbitrary constant.

Train/val/test split ratios come from config.yaml's data.* keys via
src/utils/data_split.py, shared with other training scripts. A genuine
held-out test split (never touched during training or the periodic val
checks) is scored once at the end with src/utils/metrics.classification_report
and saved alongside the checkpoint.

Usage:
    python refactoring_mining/train_refactoring_predictor.py \
        --dataset refactoring_mining/labeled_dataset.pt \
        --output models/refactoring_predictor_trained.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gnn.refactoring_types import REFACTORING_TYPE_TO_ID
from src.utils.data_split import split_indices
from src.utils.helpers import ConfigManager
from src.utils.metrics import classification_report


def _build_heads(embedding_dim: int, num_refactoring_types: int):
    """Mirrors RefactoringPredictor's classifier/confidence_predictor architecture"""
    classifier = nn.Sequential(
        nn.Linear(embedding_dim, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, num_refactoring_types)
    )
    confidence_predictor = nn.Sequential(
        nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 1), nn.Sigmoid()
    )
    return classifier, confidence_predictor


def train(dataset_path: Path, output_path: Path, config_path: str, epochs: int = 30, lr: float = 0.001):
    data = torch.load(dataset_path)
    embeddings, labels = data["embeddings"], data["labels"]
    num_examples = embeddings.shape[0]
    if num_examples < 10:
        raise ValueError(f"Only {num_examples} labeled examples - mine more repos before training")

    config = ConfigManager(config_path)
    train_idx, val_idx, test_idx = split_indices(num_examples, config)

    embedding_dim = embeddings.shape[1]
    num_types = len(REFACTORING_TYPE_TO_ID)
    classifier, confidence_predictor = _build_heads(embedding_dim, num_types)

    params = list(classifier.parameters()) + list(confidence_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # The mined dataset is heavily imbalanced (e.g. remove_dead_code has 20x more
    # examples than extract_method) - unweighted cross-entropy lets the classifier
    # ignore minority classes entirely and still get a low training loss. Plain
    # inverse-frequency weighting overcorrects here: with a class as rare as
    # extract_interface (a handful of examples), its weight becomes so large the
    # model starts predicting it constantly (high recall, near-zero precision)
    # while starving the majority classes instead. Sqrt-dampened weighting, capped
    # to a 5x spread, is a milder compromise (computed from the TRAIN split only,
    # so the held-out test set stays untouched).
    train_counts = torch.bincount(labels[train_idx], minlength=num_types).float()
    present = train_counts > 0
    inv_sqrt_freq = torch.where(present, 1.0 / train_counts.clamp(min=1).sqrt(), torch.zeros(num_types))
    mean_weight = inv_sqrt_freq[present].mean()
    class_weights = torch.where(present, (inv_sqrt_freq / mean_weight).clamp(min=0.2, max=5.0), torch.zeros(num_types))

    logger.info(f"Training on {len(train_idx)} examples, validating on {len(val_idx)}")

    for epoch in range(1, epochs + 1):
        classifier.train()
        confidence_predictor.train()
        optimizer.zero_grad()

        logits = classifier(embeddings[train_idx])
        class_loss = F.cross_entropy(logits, labels[train_idx], weight=class_weights)
        
        with torch.no_grad():
            predicted = torch.argmax(logits, dim=-1)
            correctness = (predicted == labels[train_idx]).float()
        confidence = confidence_predictor(embeddings[train_idx]).squeeze(-1)
        confidence_loss = F.binary_cross_entropy(confidence, correctness)
        
        loss = class_loss + confidence_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            classifier.eval()
            confidence_predictor.eval()
            with torch.no_grad():
                val_logits = classifier(embeddings[val_idx])
                val_acc = (torch.argmax(val_logits, dim=-1) == labels[val_idx]).float().mean().item()
            logger.info(f"epoch {epoch}/{epochs} - train loss: {loss.item():.4f} - val accuracy: {val_acc:.3f}")
    
    classifier.eval()
    confidence_predictor.eval()
    id_to_type = {v: k for k, v in REFACTORING_TYPE_TO_ID.items()}
    class_names = [id_to_type[i] for i in range(num_types)]
    with torch.no_grad():
        test_logits = classifier(embeddings[test_idx])
        test_acc = (torch.argmax(test_logits, dim=-1) == labels[test_idx]).float().mean().item()
        test_report = classification_report(test_logits, labels[test_idx], class_names)
    logger.info(f"Held-out test accuracy: {test_acc:.3f} "
                f"(macro F1: {test_report['macro']['f1']:.3f}, n={len(test_idx)})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "classifier_state": classifier.state_dict(),
        "confidence_predictor_state": confidence_predictor.state_dict(),
        "embedding_dim": embedding_dim,
        "num_refactoring_types": num_types
    }, output_path)
    logger.info(f"Saved trained RefactoringPredictor heads to {output_path}")

    metrics_path = output_path.parent / (output_path.stem + ".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "dataset": str(dataset_path),
            "num_examples": num_examples,
            "train_size": len(train_idx), "val_size": len(val_idx), "test_size": len(test_idx),
            "test_accuracy": test_acc,
            "test_classification_report": test_report,
        }, f, indent=2)
    logger.info(f"Saved test-set classification report to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RefactoringPredictor on mined refactoring labels")
    parser.add_argument("--dataset", default="./refactoring_mining/labeled_dataset.pt")
    parser.add_argument("--output", default="./models/refactoring_predictor_trained.pt")
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(Path(args.dataset), Path(args.output), args.config, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
