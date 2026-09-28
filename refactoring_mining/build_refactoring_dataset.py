"""
Turns RefactoringMiner's mined JSON into a labeled (embedding, refactoring_type)
dataset for training RefactoringPredictor.

For each detected refactoring we check out the PARENT commit (the "before"
state RefactoringMiner's leftSideLocations describe), build the CPG for the
repo at that commit, best-effort match the refactored code element to a CPG
node by file path + name, and record that node's GNN embedding with the
mapped label. Refactorings within the same commit are batched so the CPG is
only built once per commit.

Usage:
    python refactoring_mining/build_refactoring_dataset.py \
        --mined-json refactoring_mining/mined_json/ErikHage_MicroBreweryModel.json \
        --repo-path refactoring_mining/repos/ErikHage_MicroBreweryModel \
        --output refactoring_mining/labeled_dataset.pt
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gnn.gnn_model import DependencyHypergraphEncoder, HypergraphGNN, convert_cpg_to_geometric_data
from src.gnn.refactoring_types import map_rm_type_to_id
from src.cpg.cpg_builder import CodePropertyGraph

NODE_TYPE_TO_ID = {
    "class": 0, "method": 1, "function": 2, "field": 3,
    "variable": 4, "statement": 5, "call": 6, "definition": 7
}
_PREFERRED_ELEMENT_TYPES = ("METHOD_DECLARATION", "TYPE_DECLARATION")


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _git_parent_sha(repo_path: Path, sha1: str) -> str:
    result = _run(["git", "rev-parse", f"{sha1}^"], repo_path)
    return result.stdout.strip() if result.returncode == 0 else None


def _git_checkout(repo_path: Path, sha1: str) -> bool:
    result = _run(["git", "checkout", "--force", sha1], repo_path)
    return result.returncode == 0


def _pick_location(refactoring: dict):
    """Prefer a method/class-level location so it matches our CPG node granularity"""
    locations = refactoring.get("leftSideLocations", [])
    for loc in locations:
        if loc.get("codeElementType") in _PREFERRED_ELEMENT_TYPES:
            return loc
    return locations[0] if locations else None


def _find_matching_node(cpg: CodePropertyGraph, file_path: str, code_element: str):
    """Best-effort match: file path suffix + node name appearing in the code element string"""
    normalized_file = file_path.replace("/", "\\")
    candidates = []
    for node in cpg.nodes.values():
        if not node.file_path.endswith(normalized_file) and not node.file_path.endswith(file_path):
            continue
        if node.name and node.name in code_element:
            candidates.append(node)
    # Prefer the most specific (longest) name match
    candidates.sort(key=lambda n: len(n.name or ""), reverse=True)
    return candidates[0] if candidates else None


def build_dataset(mined_json_path: Path, repo_path: Path, language: str = "java",
                   node_embedding_dim: int = 64, hidden_dims=None, output_dim: int = 128,
                   max_refactorings: int = None):
    hidden_dims = hidden_dims or [256, 256]
    
    with open(mined_json_path) as f:
        mined = json.load(f)
    
    # Group refactorings by parent commit so we only rebuild the CPG once per commit
    by_parent_commit = defaultdict(list)
    for commit in mined.get("commits", []):
        sha1 = commit.get("sha1")
        parent_sha = _git_parent_sha(repo_path, sha1)
        if not parent_sha:
            continue
        for refactoring in commit.get("refactorings", []):
            label_id = map_rm_type_to_id(refactoring.get("type", ""))
            if label_id is None:
                continue
            location = _pick_location(refactoring)
            if not location:
                continue
            by_parent_commit[parent_sha].append((location, label_id, refactoring["type"]))
    
    if max_refactorings:
        flattened = [(sha, item) for sha, items in by_parent_commit.items() for item in items][:max_refactorings]
        by_parent_commit = defaultdict(list)
        for sha, item in flattened:
            by_parent_commit[sha].append(item)
    
    logger.info(f"{len(by_parent_commit)} unique parent commits to inspect, "
                f"{sum(len(v) for v in by_parent_commit.values())} candidate refactorings")
    
    node_encoder = DependencyHypergraphEncoder(node_type_vocab_size=len(NODE_TYPE_TO_ID), embedding_dim=node_embedding_dim)
    gnn_model = HypergraphGNN(input_dim=node_embedding_dim, hidden_dims=hidden_dims, output_dim=output_dim, model_type="gat")
    node_encoder.eval()
    gnn_model.eval()
    
    original_branch_result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_path)
    original_ref = original_branch_result.stdout.strip() or "HEAD"
    
    embeddings, labels, matched, skipped = [], [], 0, 0
    
    try:
        for parent_sha, items in by_parent_commit.items():
            if not _git_checkout(repo_path, parent_sha):
                logger.warning(f"Could not checkout {parent_sha}, skipping {len(items)} refactorings")
                skipped += len(items)
                continue
            
            cpg = CodePropertyGraph(language=language)
            cpg.build_from_directory(str(repo_path))
            if len(cpg.nodes) == 0 or len(cpg.edges) == 0:
                skipped += len(items)
                continue
            
            cpg_dict = cpg.to_dict()
            data = convert_cpg_to_geometric_data(cpg_dict, NODE_TYPE_TO_ID, encoder=node_encoder)
            # Matches the insertion-order row indexing used inside convert_cpg_to_geometric_data
            node_id_order = list(cpg_dict["nodes"].keys())
            
            with torch.no_grad():
                node_embeddings, _ = gnn_model(data.x.float(), data.edge_index)
            
            for location, label_id, rm_type in items:
                node = _find_matching_node(cpg, location["filePath"], location.get("codeElement", ""))
                if node is None:
                    skipped += 1
                    continue
                try:
                    row = node_id_order.index(node.node_id)
                except ValueError:
                    skipped += 1
                    continue
                embeddings.append(node_embeddings[row].detach())
                labels.append(label_id)
                matched += 1
    finally:
        _git_checkout(repo_path, original_ref)
    
    logger.info(f"Matched {matched} labeled examples, skipped {skipped}")
    return embeddings, labels


def main():
    parser = argparse.ArgumentParser(description="Build a labeled refactoring-type dataset from mined JSON")
    parser.add_argument("--mined-json", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--output", default="./refactoring_mining/labeled_dataset.pt")
    parser.add_argument("--max-refactorings", type=int, default=None)
    args = parser.parse_args()
    
    embeddings, labels = build_dataset(
        Path(args.mined_json), Path(args.repo_path), max_refactorings=args.max_refactorings
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
