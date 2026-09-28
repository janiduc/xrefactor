"""
Turns RefactoringMiner's mined JSON into a labeled (embedding, refactoring_type)
dataset for training RefactoringPredictor.

For each detected refactoring we check out the PARENT commit (the "before"
state RefactoringMiner's leftSideLocations describe), build the CPG for the
repo at that commit, best-effort match the refactored code element to a CPG
node by file path + name, and record that node's GNN embedding with the
mapped label. Refactorings within the same commit are batched so the CPG is
only built once per commit.

Embeddings are computed by a node encoder + GNN that, by default, are
randomly initialized - exactly like a real pipeline run with no
gnn.pretrained_checkpoint configured. Pass --gnn-checkpoint (a checkpoint
produced by `python -m src.gnn.pretrain`, same format pipeline.py loads) to
compute embeddings with the same weights the real pipeline will use at
inference time. Rebuild this dataset whenever that checkpoint changes -
embeddings from different encoder weights are not comparable.

Usage:
    python refactoring_mining/build_refactoring_dataset.py \
        --mined-json refactoring_mining/mined_json/ErikHage_MicroBreweryModel.json \
        --repo-path refactoring_mining/repos/ErikHage_MicroBreweryModel \
        --gnn-checkpoint models/gnn_pretrained.pt \
        --output refactoring_mining/labeled_dataset.pt
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gnn.gnn_model import DependencyHypergraphEncoder, HypergraphGNN, convert_cpg_to_geometric_data
from src.gnn.refactoring_types import map_rm_type_to_id
from src.cpg.cpg_builder import CodeNode, CodePropertyGraph

NODE_TYPE_TO_ID = {
    "class": 0, "method": 1, "function": 2, "field": 3,
    "variable": 4, "statement": 5, "call": 6, "definition": 7
}
_PREFERRED_ELEMENT_TYPES = ("METHOD_DECLARATION", "TYPE_DECLARATION")


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _git_parent_sha(repo_path: Path, sha1: str) -> Optional[str]:
    result = _run(["git", "rev-parse", f"{sha1}^"], repo_path)
    return result.stdout.strip() if result.returncode == 0 else None


def _git_checkout(repo_path: Path, sha1: str) -> bool:
    """git checkout exits non-zero if it fails to write ANY file - on Windows this
    routinely happens for a handful of long-named test-fixture files (>260 char
    paths) even when the working tree otherwise switched correctly. Since we only
    ever read .java files afterward, treat the checkout as usable whenever HEAD
    actually moved to the target commit, rather than trusting the exit code."""
    _run(["git", "checkout", "--force", sha1], repo_path)
    head = _run(["git", "rev-parse", "HEAD"], repo_path)
    return head.stdout.strip() == sha1


def _candidate_locations(refactoring: dict) -> List[dict]:
    """All leftSideLocations, ordered method/class-level declarations first
    (to match our CPG node granularity), so the caller can try each in turn
    instead of committing to a single arbitrary pick."""
    locations = refactoring.get("leftSideLocations", [])
    preferred = [loc for loc in locations if loc.get("codeElementType") in _PREFERRED_ELEMENT_TYPES]
    rest = [loc for loc in locations if loc.get("codeElementType") not in _PREFERRED_ELEMENT_TYPES]
    return preferred + rest


def _normalize_path(path: str) -> str:
    return Path(path).as_posix()


def _path_matches(node_file_path: str, rm_file_path: str) -> bool:
    node_posix = _normalize_path(node_file_path)
    rm_posix = _normalize_path(rm_file_path)
    return node_posix == rm_posix or node_posix.endswith("/" + rm_posix)


def _name_match_kind(node_name: str, code_element: str) -> Optional[str]:
    """Returns "exact", "fallback", or None. Exact = node name equals the last
    dotted segment of the qualified code element (stripped of any parameter
    list). Fallback = node name merely appears as a substring - kept as a
    last resort since it can false-match short/common names."""
    if not node_name or not code_element:
        return None
    last_segment = code_element.rsplit(".", 1)[-1].split("(")[0]
    if node_name == last_segment:
        return "exact"
    if node_name in code_element:
        return "fallback"
    return None


def _find_matching_node(
    cpg: CodePropertyGraph, locations: List[dict]
) -> Tuple[Optional[CodeNode], bool]:
    """Try each candidate location in preference order; within a location,
    prefer an exact name match over a substring-fallback match, and the
    longest node name among ties. Returns (node_or_None, used_fallback)."""
    for location in locations:
        file_path = location.get("filePath", "")
        code_element = location.get("codeElement", "")

        path_matched = [node for node in cpg.nodes.values() if _path_matches(node.file_path, file_path)]
        if not path_matched:
            continue

        exact_candidates, fallback_candidates = [], []
        for node in path_matched:
            kind = _name_match_kind(node.name, code_element)
            if kind == "exact":
                exact_candidates.append(node)
            elif kind == "fallback":
                fallback_candidates.append(node)

        if exact_candidates:
            exact_candidates.sort(key=lambda n: len(n.name or ""), reverse=True)
            return exact_candidates[0], False

        if fallback_candidates:
            fallback_candidates.sort(key=lambda n: len(n.name or ""), reverse=True)
            logger.debug(f"Using fallback substring name match for '{code_element}' in {file_path}")
            return fallback_candidates[0], True

    return None, False


def _load_gnn_checkpoint_if_available(node_encoder, gnn_model, checkpoint_path: Optional[str], device: str) -> None:
    """Mirrors XRefactorPipeline._load_gnn_checkpoint_if_available so training
    embeddings and real-pipeline embeddings come from the same weights."""
    if not checkpoint_path:
        logger.warning(
            "No --gnn-checkpoint given - node encoder/GNN weights are randomly "
            "initialized. The resulting embeddings will not match what a real "
            "pipeline run using gnn.pretrained_checkpoint would produce."
        )
        return

    if not Path(checkpoint_path).exists():
        logger.warning(f"--gnn-checkpoint '{checkpoint_path}' not found - using random init")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    node_encoder.load_state_dict(checkpoint["node_encoder_state"])
    gnn_model.load_state_dict(checkpoint["gnn_model_state"])
    logger.info(f"Loaded pretrained GNN weights from {checkpoint_path}")


def build_dataset(mined_json_path: Path, repo_path: Path, language: str = "java",
                   node_embedding_dim: int = 64, hidden_dims=None, output_dim: int = 128,
                   max_refactorings: int = None, gnn_checkpoint: Optional[str] = None,
                   device: str = "cpu"):
    hidden_dims = hidden_dims or [256, 256]

    with open(mined_json_path, encoding="utf-8") as f:
        mined = json.load(f)

    # Group refactorings by parent commit so we only rebuild the CPG once per commit
    by_parent_commit = defaultdict(list)
    skip_counts = {
        "no_parent": 0, "no_location": 0, "checkout_failed": 0,
        "empty_cpg": 0, "no_node_match": 0, "row_lookup_failed": 0,
    }

    for commit in mined.get("commits", []):
        sha1 = commit.get("sha1")
        parent_sha = _git_parent_sha(repo_path, sha1)
        refactorings = commit.get("refactorings", [])
        if not parent_sha:
            skip_counts["no_parent"] += len(refactorings)
            continue
        for refactoring in refactorings:
            label_id = map_rm_type_to_id(refactoring.get("type", ""))
            if label_id is None:
                continue
            locations = _candidate_locations(refactoring)
            if not locations:
                skip_counts["no_location"] += 1
                continue
            by_parent_commit[parent_sha].append((locations, label_id, refactoring["type"]))

    if max_refactorings:
        flattened = [(sha, item) for sha, items in by_parent_commit.items() for item in items][:max_refactorings]
        by_parent_commit = defaultdict(list)
        for sha, item in flattened:
            by_parent_commit[sha].append(item)

    logger.info(f"{len(by_parent_commit)} unique parent commits to inspect, "
                f"{sum(len(v) for v in by_parent_commit.values())} candidate refactorings")

    node_encoder = DependencyHypergraphEncoder(node_type_vocab_size=len(NODE_TYPE_TO_ID), embedding_dim=node_embedding_dim)
    gnn_model = HypergraphGNN(input_dim=node_embedding_dim, hidden_dims=hidden_dims, output_dim=output_dim, model_type="gat")
    _load_gnn_checkpoint_if_available(node_encoder, gnn_model, gnn_checkpoint, device)
    node_encoder.eval()
    gnn_model.eval()

    original_branch_result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_path)
    original_ref = original_branch_result.stdout.strip() or "HEAD"

    embeddings, labels, matched, fallback_matches = [], [], 0, 0

    try:
        for parent_sha, items in by_parent_commit.items():
            if not _git_checkout(repo_path, parent_sha):
                logger.warning(f"Could not checkout {parent_sha}, skipping {len(items)} refactorings")
                skip_counts["checkout_failed"] += len(items)
                continue

            cpg = CodePropertyGraph(language=language)
            cpg.build_from_directory(str(repo_path))
            if len(cpg.nodes) == 0 or len(cpg.edges) == 0:
                skip_counts["empty_cpg"] += len(items)
                continue

            cpg_dict = cpg.to_dict()
            data = convert_cpg_to_geometric_data(cpg_dict, NODE_TYPE_TO_ID, encoder=node_encoder)
            # Matches the insertion-order row indexing used inside convert_cpg_to_geometric_data
            node_id_order = list(cpg_dict["nodes"].keys())

            with torch.no_grad():
                node_embeddings, _ = gnn_model(data.x.float(), data.edge_index)

            for locations, label_id, rm_type in items:
                node, used_fallback = _find_matching_node(cpg, locations)
                if node is None:
                    skip_counts["no_node_match"] += 1
                    continue
                try:
                    row = node_id_order.index(node.node_id)
                except ValueError:
                    skip_counts["row_lookup_failed"] += 1
                    continue
                embeddings.append(node_embeddings[row].detach())
                labels.append(label_id)
                matched += 1
                if used_fallback:
                    fallback_matches += 1
    finally:
        _git_checkout(repo_path, original_ref)

    total_skipped = sum(skip_counts.values())
    logger.info(f"Matched {matched} labeled examples ({fallback_matches} via fallback name matching), "
                f"skipped {total_skipped}")
    for reason, count in skip_counts.items():
        if count:
            logger.info(f"  skipped ({reason}): {count}")

    return embeddings, labels


def main():
    parser = argparse.ArgumentParser(description="Build a labeled refactoring-type dataset from mined JSON")
    parser.add_argument("--mined-json", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--output", default="./refactoring_mining/labeled_dataset.pt")
    parser.add_argument("--max-refactorings", type=int, default=None)
    parser.add_argument("--gnn-checkpoint", default=None,
                         help="Path to a checkpoint from `python -m src.gnn.pretrain`. "
                              "Without this, embeddings use a randomly-initialized GNN.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    embeddings, labels = build_dataset(
        Path(args.mined_json), Path(args.repo_path), max_refactorings=args.max_refactorings,
        gnn_checkpoint=args.gnn_checkpoint, device=args.device,
    )

    if not embeddings:
        logger.warning("No labeled examples were produced - nothing saved")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {"embeddings": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
    if output_path.exists():
        existing = torch.load(output_path)

    combined_embeddings = torch.cat([existing["embeddings"], torch.stack(embeddings)]) if existing["embeddings"].numel() else torch.stack(embeddings)
    combined_labels = torch.cat([existing["labels"], torch.tensor(labels, dtype=torch.long)]) if existing["labels"].numel() else torch.tensor(labels, dtype=torch.long)

    torch.save({"embeddings": combined_embeddings, "labels": combined_labels}, output_path)
    logger.info(f"Saved {combined_embeddings.shape[0]} total labeled examples to {output_path}")


if __name__ == "__main__":
    main()
