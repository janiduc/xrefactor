"""
XRefactor Evaluation Harness

Produces the numbers for the research writeup, re-runnable end to end:
1. Stage 2 classifier: precision/recall/F1 on labeled_dataset.pt's held-out test split.
2. Stage 3 transformer: BLEU/edit-distance/exact-match AND structural pass rate
   (src/transformer/pattern_validators.py), reported PER PATTERN - the headline
   number, since the project's core requirement is correctly-refactored code
   per identified pattern, not just plausible-looking text.
3. Stage 4 XAI: the problem/solution-score separation check from Phase 6's
   weak supervision (no invented ground-truth metric - see the report for why).
4. End-to-end smell-to-refactor traceability: runs the full pipeline on a
   held-out repo never used in mining/training, tracing detected smell ->
   predicted pattern -> generated code -> structural validator result for a
   bounded sample of suggestions.

Usage:
    python evaluate.py --config ./configs/config.yaml --output ./outputs
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from loguru import logger

from src.core.pipeline import XRefactorPipeline
from src.gnn.gnn_model import HypergraphGNN, RefactoringPredictor
from src.gnn.refactoring_types import REFACTORING_TYPE_TO_ID
from src.transformer.code_generator import CodeTransformer
from src.transformer.pattern_validators import validate
from src.utils.data_split import split_indices
from src.utils.helpers import ConfigManager
from src.utils.metrics import aggregate_generation_quality, classification_report, generation_quality
from src.xai.explanation_module import CausalInferenceModule

ID_TO_TYPE = {v: k for k, v in REFACTORING_TYPE_TO_ID.items()}


def evaluate_classifier(config: ConfigManager) -> dict:
    """Stage 2: fresh forward pass of the trained predictor over its held-out test split."""
    dataset_path = "./refactoring_mining/labeled_dataset.pt"
    checkpoint_path = config.get("gnn.refactoring_predictor_checkpoint")
    if not os.path.exists(dataset_path) or not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"skipped": True, "reason": "labeled_dataset.pt or refactoring_predictor checkpoint not found"}

    data = torch.load(dataset_path)
    embeddings, labels = data["embeddings"], data["labels"]
    _, _, test_idx = split_indices(embeddings.shape[0], config)

    gnn = HypergraphGNN(input_dim=64, hidden_dims=[256, 256], output_dim=embeddings.shape[1], model_type="gat")
    predictor = RefactoringPredictor(gnn, output_dim=embeddings.shape[1], num_refactoring_types=len(REFACTORING_TYPE_TO_ID))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    predictor.classifier.load_state_dict(checkpoint["classifier_state"])
    predictor.confidence_predictor.load_state_dict(checkpoint["confidence_predictor_state"])
    predictor.eval()

    with torch.no_grad():
        logits = predictor.classifier(embeddings[test_idx])
    class_names = [ID_TO_TYPE[i] for i in range(len(REFACTORING_TYPE_TO_ID))]
    report = classification_report(logits, labels[test_idx], class_names)
    report["test_size"] = len(test_idx)
    return report


def evaluate_transformer(config: ConfigManager, eval_sample_size: int, eval_max_length: int) -> dict:
    """Stage 3: per-pattern generation quality AND structural pass rate on a held-out sample."""
    dataset_path = "./refactoring_mining/codegen_pairs.jsonl"
    checkpoint_path = config.get("transformer.pretrained_checkpoint")
    if not os.path.exists(dataset_path) or not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"skipped": True, "reason": "codegen_pairs.jsonl or transformer checkpoint not found"}

    pairs = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    _, _, test_idx = split_indices(len(pairs), config)
    test_idx = test_idx.tolist()[:eval_sample_size]

    model = CodeTransformer(model_name="microsoft/codebert-base")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.decoder.load_state_dict(checkpoint["decoder_state"])
    model.fusion_layer.load_state_dict(checkpoint["fusion_layer_state"])
    model.gnn_projection.load_state_dict(checkpoint["gnn_projection_state"])
    model.eval()

    gnn_embedding_dim = checkpoint.get("gnn_embedding_dim", 128)
    per_type_results = {}
    for i in test_idx:
        pair = pairs[i]
        type_name = ID_TO_TYPE[pair["refactoring_type_id"]]
        # Same documented placeholder as training: no real per-example GNN embedding
        # is cheaply available here either, so a fixed neutral vector stands in.
        gnn_emb = torch.zeros(1, gnn_embedding_dim)
        with torch.no_grad():
            generated = model.generate(source_code=pair["before"], gnn_embeddings=gnn_emb,
                                        refactoring_type=pair["refactoring_type_id"], max_length=eval_max_length)
        quality = generation_quality(pair["after"], generated)
        validation = validate(type_name, pair["before"], generated)
        bucket = per_type_results.setdefault(type_name, {"quality": [], "structural_passed": 0, "count": 0})
        bucket["quality"].append({"reference": pair["after"], "hypothesis": generated})
        bucket["structural_passed"] += int(validation["passed"])
        bucket["count"] += 1

    per_type_report = {}
    total_passed, total_count = 0, 0
    for type_name, bucket in per_type_results.items():
        agg = aggregate_generation_quality(bucket["quality"])
        pass_rate = bucket["structural_passed"] / bucket["count"]
        per_type_report[type_name] = {**agg, "structural_pass_rate": pass_rate}
        total_passed += bucket["structural_passed"]
        total_count += bucket["count"]

    return {
        "eval_sample_size": len(test_idx),
        "overall_structural_pass_rate": (total_passed / total_count) if total_count else None,
        "per_pattern": per_type_report,
    }


def evaluate_xai(config: ConfigManager) -> dict:
    """Stage 4: problem/solution-score separation check from Phase 6's weak supervision."""
    cache_path = "./refactoring_mining/causal_dataset_cache.pt"
    checkpoint_path = config.get("xai.pretrained_checkpoint")
    if not os.path.exists(cache_path) or not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"skipped": True, "reason": "causal_dataset_cache.pt or xai checkpoint not found"}

    data = torch.load(cache_path)
    problem_embeddings, problem_labels = data["problem_embeddings"], data["problem_labels"]
    before_embeddings, after_embeddings, solution_labels = (
        data["before_embeddings"], data["after_embeddings"], data["solution_labels"]
    )
    _, _, p_test_idx = split_indices(problem_embeddings.shape[0], config)
    _, _, s_test_idx = split_indices(before_embeddings.shape[0], config)

    module = CausalInferenceModule(feature_dim=problem_embeddings.shape[1])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    module.problem_detector.load_state_dict(checkpoint["problem_detector_state"])
    module.solution_evaluator.load_state_dict(checkpoint["solution_evaluator_state"])
    module.eval()

    with torch.no_grad():
        p_scores = module.problem_detector(problem_embeddings[p_test_idx]).squeeze(-1)
        p_acc = ((p_scores > 0.5).float() == problem_labels[p_test_idx]).float().mean().item()
        mean_score_positive = p_scores[problem_labels[p_test_idx] == 1].mean().item() if (problem_labels[p_test_idx] == 1).any() else None
        mean_score_negative = p_scores[problem_labels[p_test_idx] == 0].mean().item() if (problem_labels[p_test_idx] == 0).any() else None

        combined = torch.cat([before_embeddings[s_test_idx], after_embeddings[s_test_idx]], dim=-1)
        s_scores = module.solution_evaluator(combined).squeeze(-1)
        s_acc = ((s_scores > 0.5).float() == solution_labels[s_test_idx]).float().mean().item()

    return {
        "note": "Weak/proxy supervision - no ground-truth 'problem severity' or 'fix quality' "
                "labels exist. This reports separation on the same kind of proxy signal used in "
                "training (see src/xai/train_causal_module.py), not validated ground truth.",
        "problem_detector_test_accuracy": p_acc,
        "problem_detector_mean_score_on_known_problems": mean_score_positive,
        "problem_detector_mean_score_on_random_nodes": mean_score_negative,
        "solution_evaluator_test_accuracy": s_acc,
        "recommendation": "Report a small human rubric (1-5 scale, N sampled evidence cards: "
                           "does the explanation match the real diff? does confidence track "
                           "whether the suggestion looks sensible?) alongside these numbers - "
                           "no automatic metric substitutes for that judgment here.",
    }


def evaluate_end_to_end(config_path: str, held_out_repo: str, top_k: int, eval_max_length: int) -> dict:
    """Full pipeline on a repo never used in mining/training: trace detected smell ->
    predicted pattern -> generated code -> structural validator result."""
    pipeline = XRefactorPipeline(config_path=config_path, device="cpu")
    cpg = pipeline.stage_1_cpg_construction(held_out_repo)
    gnn_model, node_embeddings = pipeline.stage_2_gnn_reasoning(cpg)

    cpg_dict = cpg.to_dict()
    samples = []
    for node_index, (node_id, node) in enumerate(cpg_dict["nodes"].items()):
        if node.get("type") == "method" and node.get("code_snippet"):
            samples.append({"code_snippet": node["code_snippet"], "file": node.get("file"),
                             "type": node.get("type"), "node_index": node_index, "node_id": node_id})
        if len(samples) >= top_k:
            break

    suggestions = pipeline.stage_3_transformer_generation(samples, node_embeddings)

    traced = []
    structural_passed = 0
    for suggestion in suggestions:
        validation = suggestion.get("structural_validation", {})
        traced.append({
            "detected_smell_pattern": suggestion.get("detected_smell_pattern"),
            "learned_pattern": suggestion.get("learned_pattern"),
            "pattern_agreement": suggestion.get("pattern_agreement"),
            "final_pattern_used": suggestion.get("refactoring_type"),
            "structural_validation_passed": validation.get("passed"),
            "structural_validation_reason": validation.get("reason"),
            "confidence": suggestion.get("confidence"),
        })
        structural_passed += int(bool(validation.get("passed")))

    return {
        "repo": held_out_repo,
        "num_smells_detected": len(pipeline.detected_smells),
        "num_suggestions_traced": len(traced),
        "structural_pass_rate": (structural_passed / len(traced)) if traced else None,
        "traces": traced,
    }


def main():
    parser = argparse.ArgumentParser(description="XRefactor evaluation harness")
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument("--output", default="./outputs")
    parser.add_argument("--held-out-repo", default="../Data/IceIce1ce/Practice-Shopping-Android-App",
                         help="A repo never used in mining/training, for the end-to-end trace")
    parser.add_argument("--top-k", type=int, default=5, help="Suggestions to trace end-to-end")
    parser.add_argument("--transformer-eval-sample-size", type=int, default=30)
    parser.add_argument("--eval-max-length", type=int, default=40)
    parser.add_argument("--skip-end-to-end", action="store_true")
    args = parser.parse_args()

    config = ConfigManager(args.config)
    report = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")}

    logger.info("Evaluating Stage 2 classifier...")
    report["stage_2_classifier"] = evaluate_classifier(config)

    logger.info("Evaluating Stage 3 transformer (per-pattern generation quality + structural pass rate)...")
    report["stage_3_transformer"] = evaluate_transformer(
        config, args.transformer_eval_sample_size, args.eval_max_length
    )

    logger.info("Evaluating Stage 4 XAI (weak-supervision separation check)...")
    report["stage_4_xai"] = evaluate_xai(config)

    if not args.skip_end_to_end:
        logger.info(f"Running end-to-end smell-to-refactor trace on {args.held_out_repo}...")
        report["end_to_end_trace"] = evaluate_end_to_end(
            args.config, args.held_out_repo, args.top_k, args.eval_max_length
        )

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"evaluation_report_{report['timestamp']}.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Evaluation report saved to {output_path}")

    if not report["stage_3_transformer"].get("skipped"):
        logger.info(f"Stage 3 overall structural pass rate: "
                    f"{report['stage_3_transformer']['overall_structural_pass_rate']}")
    if "end_to_end_trace" in report:
        logger.info(f"End-to-end structural pass rate: {report['end_to_end_trace']['structural_pass_rate']}")


if __name__ == "__main__":
    main()
