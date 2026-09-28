"""
Supervised fine-tuning of RefactoringPredictor's classifier + confidence heads
using the labeled (embedding, refactoring_type) dataset produced by
build_refactoring_dataset.py.

The classifier is trained with standard cross-entropy against the mined
labels. The confidence head is trained to predict whether the classifier's
own top prediction was correct (a common calibration trick), rather than
being hand-set to an arbitrary constant.

Usage:
    python refactoring_mining/train_refactoring_predictor.py \
        --dataset refactoring_mining/labeled_dataset.pt \
        --output models/refactoring_predictor_trained.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gnn.refactoring_types import REFACTORING_TYPE_TO_ID


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


def train(dataset_path: Path, output_path: Path, epochs: int = 30, lr: float = 0.001,
          val_split: float = 0.15, seed: int = 42):
    data = torch.load(dataset_path)
    embeddings, labels = data["embeddings"], data["labels"]
    num_examples = embeddings.shape[0]
    if num_examples < 10:
        raise ValueError(f"Only {num_examples} labeled examples - mine more repos before training")
    
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_examples, generator=generator)
    val_size = max(1, int(num_examples * val_split))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    
    embedding_dim = embeddings.shape[1]
    num_types = len(REFACTORING_TYPE_TO_ID)
    classifier, confidence_predictor = _build_heads(embedding_dim, num_types)
    
    params = list(classifier.parameters()) + list(confidence_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    
    logger.info(f"Training on {len(train_idx)} examples, validating on {len(val_idx)}")
    
    for epoch in range(1, epochs + 1):
        classifier.train()
        confidence_predictor.train()
        optimizer.zero_grad()
        
        logits = classifier(embeddings[train_idx])
        class_loss = F.cross_entropy(logits, labels[train_idx])
        
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
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "classifier_state": classifier.state_dict(),
        "confidence_predictor_state": confidence_predictor.state_dict(),
        "embedding_dim": embedding_dim,
        "num_refactoring_types": num_types
    }, output_path)
    logger.info(f"Saved trained RefactoringPredictor heads to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RefactoringPredictor on mined refactoring labels")
    parser.add_argument("--dataset", default="./refactoring_mining/labeled_dataset.pt")
    parser.add_argument("--output", default="./models/refactoring_predictor_trained.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train(Path(args.dataset), Path(args.output), epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
