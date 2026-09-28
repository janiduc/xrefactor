"""
Weak/proxy supervision for CausalInferenceModule's problem_detector and
solution_evaluator heads.

No ground-truth "this code has problem severity X" or "this fix has quality
Y" labels exist. Instead, real historical refactorings give a WEAK, PROXY
signal, not validated ground truth:

- problem_detector: the (parent-commit) embedding of a node RefactoringMiner
  actually recorded as a refactoring target is labeled 1 ("a real problem
  existed here - someone fixed it"). A random OTHER method node from the
  same commit, not flagged by any mined refactoring, is labeled 0 ("not
  flagged" - NOT a proof the code was fine, just a defensible weak negative).
- solution_evaluator: a real (before_embedding, after_embedding) pair from an
  accepted refactoring is labeled 1 ("a genuine improvement - it happened and
  was kept"). (before_embedding, before_embedding) - i.e. no change at all -
  is labeled 0, a trivial "no change is not an improvement" negative.

This requires rebuilding a CPG at BOTH the parent commit (for "before") and
the commit itself (for "after") for each sampled refactoring - twice the
per-commit cost of build_refactoring_dataset.py's embeddings, which only
needs the parent. Deliberately bounded to a small sample (--max-pairs) across
fast repos, not the full mining corpus, to keep this tractable.

Usage:
    python -m src.xai.train_causal_module \
        --pairs refactoring_mining/MicroBreweryModel_refactorings.json:refactoring_mining/repos/MicroBreweryModel \
        --pairs refactoring_mining/mined_json/SeSac-Cloud-Backend-4_TicketingWebApplication.json:refactoring_mining/repos/SeSac-Cloud-Backend-4_TicketingWebApplication \
        --gnn-checkpoint models/gnn_pretrained.pt \
        --output models/causal_module_trained.pt
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from refactoring_mining.build_refactoring_dataset import (
    NODE_TYPE_TO_ID, _candidate_locations, _find_matching_node,
)
from refactoring_mining.git_utils import git_checkout, git_current_ref, git_parent_sha
from src.cpg.cpg_builder import CodePropertyGraph
from src.gnn.gnn_model import DependencyHypergraphEncoder, HypergraphGNN, convert_cpg_to_geometric_data
from src.gnn.refactoring_types import map_rm_type_to_id
from src.utils.data_split import split_indices
from src.utils.helpers import ConfigManager
from src.xai.explanation_module import CausalInferenceModule


def _load_gnn(node_embedding_dim: int, hidden_dims: List[int], output_dim: int,
              gnn_checkpoint: Optional[str], device: str):
    node_encoder = DependencyHypergraphEncoder(node_type_vocab_size=len(NODE_TYPE_TO_ID),
                                                embedding_dim=node_embedding_dim)
    gnn_model = HypergraphGNN(input_dim=node_embedding_dim, hidden_dims=hidden_dims,
                               output_dim=output_dim, model_type="gat")
    if gnn_checkpoint and Path(gnn_checkpoint).exists():
        checkpoint = torch.load(gnn_checkpoint, map_location=device)
        node_encoder.load_state_dict(checkpoint["node_encoder_state"])
        gnn_model.load_state_dict(checkpoint["gnn_model_state"])
        logger.info(f"Loaded pretrained GNN weights from {gnn_checkpoint}")
    else:
        logger.warning("No --gnn-checkpoint given - embeddings use a randomly-initialized GNN")
    node_encoder.eval()
    gnn_model.eval()
    return node_encoder, gnn_model


def _embed_cpg(cpg: CodePropertyGraph, node_encoder, gnn_model):
    cpg_dict = cpg.to_dict()
    data = convert_cpg_to_geometric_data(cpg_dict, NODE_TYPE_TO_ID, encoder=node_encoder)
    node_id_order = list(cpg_dict["nodes"].keys())
    with torch.no_grad():
        node_embeddings, _ = gnn_model(data.x.float(), data.edge_index)
    return node_embeddings, node_id_order


def build_causal_dataset(pair_specs: List[Tuple[Path, Path]], node_encoder, gnn_model,
                          language: str = "java", max_pairs: int = 40,
                          negatives_per_commit: int = 2, seed: int = 42) -> dict:
    rng = random.Random(seed)
    problem_embeddings, problem_labels = [], []
    before_embeddings, after_embeddings, solution_labels = [], [], []

    for mined_json_path, repo_path in pair_specs:
        if len(problem_embeddings) >= max_pairs * 2:  # positives + negatives already collected
            break
        with open(mined_json_path, encoding="utf-8") as f:
            mined = json.load(f)

        by_commit = defaultdict(list)
        for commit in mined.get("commits", []):
            sha1 = commit.get("sha1")
            for refactoring in commit.get("refactorings", []):
                if map_rm_type_to_id(refactoring.get("type", "")) is None:
                    continue
                locations = _candidate_locations(refactoring)
                if locations:
                    by_commit[sha1].append(locations)

        original_ref = git_current_ref(repo_path)
        try:
            for sha1, location_lists in by_commit.items():
                if len(problem_embeddings) >= max_pairs * 2:
                    break
                parent_sha = git_parent_sha(repo_path, sha1)
                if not parent_sha:
                    continue
                if not git_checkout(repo_path, parent_sha):
                    continue
                before_cpg = CodePropertyGraph(language=language)
                before_cpg.build_from_directory(str(repo_path))
                if not before_cpg.nodes:
                    continue
                before_node_embeddings, before_id_order = _embed_cpg(before_cpg, node_encoder, gnn_model)

                if not git_checkout(repo_path, sha1):
                    continue
                after_cpg = CodePropertyGraph(language=language)
                after_cpg.build_from_directory(str(repo_path))
                if not after_cpg.nodes:
                    continue
                after_node_embeddings, after_id_order = _embed_cpg(after_cpg, node_encoder, gnn_model)

                matched_before_ids = set()
                for locations in location_lists:
                    before_node, _ = _find_matching_node(before_cpg, locations)
                    if before_node is None or before_node.node_id not in before_id_order:
                        continue
                    # Right-side locations aren't tracked separately here (this dataset only
                    # needs SOME real "after" state of the same repo, not the specific
                    # renamed/extracted entity) - reuse the same location list against the
                    # after-commit CPG; if the exact entity moved/was renamed, matching
                    # naturally fails and this pair is skipped rather than mismatched.
                    after_node, _ = _find_matching_node(after_cpg, locations)
                    if after_node is None or after_node.node_id not in after_id_order:
                        continue

                    before_row = before_id_order.index(before_node.node_id)
                    after_row = after_id_order.index(after_node.node_id)
                    e_before = before_node_embeddings[before_row].detach()
                    e_after = after_node_embeddings[after_row].detach()

                    problem_embeddings.append(e_before)
                    problem_labels.append(1.0)
                    before_embeddings.append(e_before)
                    after_embeddings.append(e_after)
                    solution_labels.append(1.0)
                    # Trivial negative: no change at all is not an improvement.
                    before_embeddings.append(e_before)
                    after_embeddings.append(e_before)
                    solution_labels.append(0.0)

                    matched_before_ids.add(before_node.node_id)

                # Weak negatives: other method nodes in this commit never flagged.
                candidate_negative_ids = [
                    nid for nid, node in before_cpg.nodes.items()
                    if node.node_type == "method" and nid not in matched_before_ids
                ]
                rng.shuffle(candidate_negative_ids)
                for nid in candidate_negative_ids[:negatives_per_commit]:
                    if nid not in before_id_order:
                        continue
                    row = before_id_order.index(nid)
                    problem_embeddings.append(before_node_embeddings[row].detach())
                    problem_labels.append(0.0)
        finally:
            git_checkout(repo_path, original_ref)

    return {
        "problem_embeddings": torch.stack(problem_embeddings) if problem_embeddings else torch.empty(0),
        "problem_labels": torch.tensor(problem_labels) if problem_labels else torch.empty(0),
        "before_embeddings": torch.stack(before_embeddings) if before_embeddings else torch.empty(0),
        "after_embeddings": torch.stack(after_embeddings) if after_embeddings else torch.empty(0),
        "solution_labels": torch.tensor(solution_labels) if solution_labels else torch.empty(0),
    }


def train(data: dict, output_path: Path, config_path: str, epochs: int = 30, lr: float = 0.001):
    problem_embeddings, problem_labels = data["problem_embeddings"], data["problem_labels"]
    before_embeddings, after_embeddings = data["before_embeddings"], data["after_embeddings"]
    solution_labels = data["solution_labels"]
    if problem_embeddings.shape[0] < 10:
        raise ValueError(f"Only {problem_embeddings.shape[0]} problem-detector examples - "
                          f"mine more refactorings/commits before training")

    feature_dim = problem_embeddings.shape[1]
    module = CausalInferenceModule(feature_dim=feature_dim)

    config = ConfigManager(config_path)
    p_train_idx, p_val_idx, _ = split_indices(problem_embeddings.shape[0], config)
    s_train_idx, s_val_idx, _ = split_indices(before_embeddings.shape[0], config)

    optimizer = torch.optim.Adam(
        list(module.problem_detector.parameters()) + list(module.solution_evaluator.parameters()), lr=lr
    )

    logger.info(f"Training problem_detector on {len(p_train_idx)} examples, "
                f"solution_evaluator on {len(s_train_idx)} examples")

    for epoch in range(1, epochs + 1):
        module.train()
        optimizer.zero_grad()

        p_logits = module.problem_detector(problem_embeddings[p_train_idx]).squeeze(-1)
        problem_loss = F.binary_cross_entropy(p_logits, problem_labels[p_train_idx])

        combined = torch.cat([before_embeddings[s_train_idx], after_embeddings[s_train_idx]], dim=-1)
        s_logits = module.solution_evaluator(combined).squeeze(-1)
        solution_loss = F.binary_cross_entropy(s_logits, solution_labels[s_train_idx])

        loss = problem_loss + solution_loss
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            module.eval()
            with torch.no_grad():
                p_val_acc = ((module.problem_detector(problem_embeddings[p_val_idx]).squeeze(-1) > 0.5).float()
                             == problem_labels[p_val_idx]).float().mean().item() if len(p_val_idx) else float("nan")
                s_val_combined = torch.cat([before_embeddings[s_val_idx], after_embeddings[s_val_idx]], dim=-1)
                s_val_acc = ((module.solution_evaluator(s_val_combined).squeeze(-1) > 0.5).float()
                             == solution_labels[s_val_idx]).float().mean().item() if len(s_val_idx) else float("nan")
            logger.info(f"epoch {epoch}/{epochs} - loss: {loss.item():.4f} - "
                        f"problem val acc: {p_val_acc:.3f} - solution val acc: {s_val_acc:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "problem_detector_state": module.problem_detector.state_dict(),
        "solution_evaluator_state": module.solution_evaluator.state_dict(),
        "feature_dim": feature_dim,
    }, output_path)
    logger.info(f"Saved trained CausalInferenceModule heads to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Weak-supervision training for CausalInferenceModule")
    parser.add_argument("--pairs", action="append", required=True,
                         help="'mined_json_path:repo_path', repeatable")
    parser.add_argument("--gnn-checkpoint", default=None)
    parser.add_argument("--output", default="./models/causal_module_trained.pt")
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument("--dataset-cache", default=None,
                         help="Optional path to save/load the built (embedding, label) dataset, "
                              "so retraining doesn't require re-mining embeddings.")
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    if args.dataset_cache and Path(args.dataset_cache).exists():
        logger.info(f"Loading cached dataset from {args.dataset_cache}")
        data = torch.load(args.dataset_cache)
    else:
        pair_specs = []
        for spec in args.pairs:
            json_path, repo_path = spec.split(":", 1)
            pair_specs.append((Path(json_path), Path(repo_path)))

        node_encoder, gnn_model = _load_gnn(64, [256, 256], 128, args.gnn_checkpoint, "cpu")
        data = build_causal_dataset(pair_specs, node_encoder, gnn_model, max_pairs=args.max_pairs)
        logger.info(f"Built dataset: {data['problem_embeddings'].shape[0]} problem examples, "
                    f"{data['before_embeddings'].shape[0]} solution examples")
        if args.dataset_cache:
            Path(args.dataset_cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, args.dataset_cache)

    train(data, Path(args.output), args.config, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
