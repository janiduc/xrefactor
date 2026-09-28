"""
Unit tests for shared utilities: data_split and metrics.
"""

import pytest
import torch

from src.utils.data_split import split_indices
from src.utils.helpers import ConfigManager
from src.utils.metrics import aggregate_generation_quality, classification_report, generation_quality


class TestSplitIndices:
    """Tests for src.utils.data_split.split_indices"""

    def test_respects_config_ratios(self):
        config = ConfigManager("./configs/config.yaml")
        num_examples = 200
        train_idx, val_idx, test_idx = split_indices(num_examples, config)

        assert abs(len(train_idx) - num_examples * 0.7) <= 2
        assert abs(len(val_idx) - num_examples * 0.15) <= 2
        assert abs(len(test_idx) - num_examples * 0.15) <= 2

    def test_splits_are_disjoint_and_complete(self):
        config = ConfigManager("./configs/config.yaml")
        num_examples = 137
        train_idx, val_idx, test_idx = split_indices(num_examples, config)

        all_idx = torch.cat([train_idx, val_idx, test_idx])
        assert len(all_idx) == num_examples
        assert len(torch.unique(all_idx)) == num_examples

    def test_same_seed_is_reproducible(self):
        config = ConfigManager("./configs/config.yaml")
        first = split_indices(50, config)
        second = split_indices(50, config)
        for a, b in zip(first, second):
            assert torch.equal(a, b)

    def test_rejects_ratios_not_summing_to_one(self):
        config = ConfigManager("./configs/config.yaml")
        config.set("data.train_split", 0.5)
        config.set("data.val_split", 0.5)
        config.set("data.test_split", 0.5)
        with pytest.raises(ValueError):
            split_indices(100, config)

    def test_rejects_too_few_examples(self):
        config = ConfigManager("./configs/config.yaml")
        with pytest.raises(ValueError):
            split_indices(2, config)


class TestClassificationReport:
    """Tests for src.utils.metrics.classification_report"""

    def test_perfect_predictions_give_accuracy_one(self):
        labels = torch.tensor([0, 1, 2, 1, 0])
        logits = torch.zeros(5, 3)
        for i, label in enumerate(labels):
            logits[i, label] = 10.0

        report = classification_report(logits, labels, class_names=["a", "b", "c"])
        assert report["accuracy"] == pytest.approx(1.0)
        assert report["macro"]["f1"] == pytest.approx(1.0)

    def test_reports_per_class_support(self):
        labels = torch.tensor([0, 0, 1])
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        report = classification_report(logits, labels, class_names=["a", "b"])
        assert report["per_class"]["a"]["support"] == 2
        assert report["per_class"]["b"]["support"] == 1


class TestGenerationQuality:
    """Tests for src.utils.metrics.generation_quality"""

    def test_identical_text_is_exact_match(self):
        result = generation_quality("public void foo() {}", "public void foo() {}")
        assert result["exact_match"] == 1.0
        assert result["edit_distance_ratio"] == pytest.approx(1.0)

    def test_completely_different_text_is_not_exact_match(self):
        result = generation_quality("public void foo() {}", "totally different tokens here")
        assert result["exact_match"] == 0.0

    def test_empty_hypothesis_scores_zero_bleu(self):
        result = generation_quality("public void foo() {}", "")
        assert result["bleu4"] == 0.0

    def test_aggregate_averages_across_pairs(self):
        pairs = [
            {"reference": "a b c", "hypothesis": "a b c"},
            {"reference": "a b c", "hypothesis": "x y z"},
        ]
        agg = aggregate_generation_quality(pairs)
        assert agg["count"] == 2
        assert agg["exact_match"] == pytest.approx(0.5)

    def test_aggregate_empty_list(self):
        agg = aggregate_generation_quality([])
        assert agg["count"] == 0
