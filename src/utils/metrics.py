"""
Shared evaluation metrics for classification and code-generation quality.

Used by refactoring_mining/train_refactoring_predictor.py,
src/transformer/train_transformer.py, and scripts/evaluate.py so every
training/eval script reports numbers the same way instead of each inventing
its own scalar accuracy calculation.
"""

from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence

import torch
from sklearn.metrics import precision_recall_fscore_support


def classification_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Per-class precision/recall/F1 plus macro and weighted averages.

    Args:
        logits: [num_examples, num_classes] raw model outputs (pre-softmax is fine).
        labels: [num_examples] integer class ids.
        class_names: names for each class id, in id order.

    Returns:
        {
            "per_class": {class_name: {"precision", "recall", "f1", "support"}},
            "macro": {"precision", "recall", "f1"},
            "weighted": {"precision", "recall", "f1"},
            "accuracy": float,
        }
    """
    predicted = torch.argmax(logits, dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    num_classes = len(class_names)

    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        labels_np, predicted, labels=list(range(num_classes)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels_np, predicted, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels_np, predicted, labels=list(range(num_classes)), average="weighted", zero_division=0
    )

    per_class = {
        class_names[i]: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    }

    return {
        "per_class": per_class,
        "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f1)},
        "weighted": {"precision": float(weighted_p), "recall": float(weighted_r), "f1": float(weighted_f1)},
        "accuracy": float((predicted == labels_np).mean()) if len(labels_np) else 0.0,
    }


def _bleu4(reference: str, hypothesis: str) -> float:
    """BLEU-4 via nltk (already a project dependency). Returns 0.0 on empty input."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not hyp_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing))


def generation_quality(reference: str, hypothesis: str) -> Dict[str, float]:
    """Cheap, dependency-light code-generation quality metrics for one (reference, hypothesis) pair.

    Note: BLEU/edit-distance measure textual similarity only, not whether the
    generated code is a structurally-correct instance of the intended
    refactoring pattern - see src/transformer/pattern_validators.py for that.
    A future improvement would be CodeBLEU (syntax + dataflow aware), not
    implemented here to avoid an extra heavyweight dependency.
    """
    exact_match = 1.0 if reference.strip() == hypothesis.strip() else 0.0
    edit_ratio = SequenceMatcher(None, reference, hypothesis).ratio()
    bleu = _bleu4(reference, hypothesis)
    return {"exact_match": exact_match, "edit_distance_ratio": edit_ratio, "bleu4": bleu}


def aggregate_generation_quality(pairs: List[Dict[str, str]]) -> Dict[str, float]:
    """Average generation_quality over a list of {"reference", "hypothesis"} dicts."""
    if not pairs:
        return {"exact_match": 0.0, "edit_distance_ratio": 0.0, "bleu4": 0.0, "count": 0}

    totals = {"exact_match": 0.0, "edit_distance_ratio": 0.0, "bleu4": 0.0}
    for pair in pairs:
        scores = generation_quality(pair["reference"], pair["hypothesis"])
        for key in totals:
            totals[key] += scores[key]

    n = len(pairs)
    result = {key: value / n for key, value in totals.items()}
    result["count"] = n
    return result
