"""
Turns RefactoringMiner's mined JSON into a before/after SOURCE TEXT dataset
for training the Stage-3 transformer (src/transformer/train_transformer.py).

This is a different extraction pass over the same mined JSON/cloned repos
that build_refactoring_dataset.py uses, producing a complementary artifact:
build_refactoring_dataset.py records (embedding, label) pairs with no
source text; this script records (before_text, after_text, label) pairs
with no embeddings. Training the transformer needs the former's GNN
context and the latter's actual code - src/transformer/train_transformer.py
combines both by embedding the same method via a CPG rebuilt at the same
commit (see that script's docstring for the tradeoffs of doing that per
example vs. once per commit).

Convention for "before" and "after": for each refactoring, we take its
preferred METHOD_DECLARATION/TYPE_DECLARATION location on the LEFT side
(the pre-refactoring commit's parent, matching build_refactoring_dataset.py's
convention) as "before", and the preferred location on the RIGHT side (the
commit itself) as "after". For most refactoring types both sides describe
the same logical entity before/after the change (e.g. Rename Variable,
Simplify Condition). For a minority of types - notably Extract Method -
RefactoringMiner's right side is the NEWLY CREATED entity (the extracted
method), not the modified original; we still pair them this way for
consistency across all 10 patterns, and document this explicitly rather
than special-casing it. This means "extract_method" training pairs teach
the model "given a method, what would a plausibly-extracted sibling method
look like" more than "given a method, shorten it" - a real, visible
simplification, not a hidden one.

Reads directly from git blobs via `git show <sha>:<path>` (no checkout
needed - cheaper than build_refactoring_dataset.py's approach, since no
live CPG is required here, just raw file text).

Usage:
    python refactoring_mining/build_codegen_dataset.py \
        --mined-json refactoring_mining/mined_json/4pxzhou_FEBS-Cloud.json \
        --repo-path refactoring_mining/repos/4pxzhou_FEBS-Cloud \
        --repo-name 4pxzhou/FEBS-Cloud \
        --output refactoring_mining/codegen_pairs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gnn.refactoring_types import map_rm_type_to_id
from refactoring_mining.git_utils import git_parent_sha, git_show_file, slice_lines

_PREFERRED_ELEMENT_TYPES = ("METHOD_DECLARATION", "TYPE_DECLARATION")


def _preferred_location(locations: List[dict]) -> Optional[dict]:
    for loc in locations:
        if loc.get("codeElementType") in _PREFERRED_ELEMENT_TYPES:
            return loc
    return locations[0] if locations else None


def _extract_text(repo_path: Path, sha: str, location: dict) -> Optional[str]:
    content = git_show_file(repo_path, sha, location.get("filePath", ""))
    if content is None:
        return None
    return slice_lines(content, location.get("startLine", 0), location.get("endLine", 0))


def build_pairs(mined_json_path: Path, repo_path: Path, repo_name: str,
                max_refactorings: Optional[int] = None) -> List[dict]:
    with open(mined_json_path, encoding="utf-8") as f:
        mined = json.load(f)

    pairs: List[dict] = []
    skipped = {"no_type_mapping": 0, "no_location": 0, "no_parent": 0, "extract_failed": 0, "no_op": 0}

    for commit in mined.get("commits", []):
        sha1 = commit.get("sha1")
        for refactoring in commit.get("refactorings", []):
            label_id = map_rm_type_to_id(refactoring.get("type", ""))
            if label_id is None:
                skipped["no_type_mapping"] += 1
                continue

            left_loc = _preferred_location(refactoring.get("leftSideLocations", []))
            right_loc = _preferred_location(refactoring.get("rightSideLocations", []))
            if not left_loc or not right_loc:
                skipped["no_location"] += 1
                continue

            parent_sha = git_parent_sha(repo_path, sha1)
            if not parent_sha:
                skipped["no_parent"] += 1
                continue

            before_text = _extract_text(repo_path, parent_sha, left_loc)
            after_text = _extract_text(repo_path, sha1, right_loc)
            if before_text is None or after_text is None:
                skipped["extract_failed"] += 1
                continue
            if before_text.strip() == after_text.strip():
                skipped["no_op"] += 1
                continue

            pairs.append({
                "before": before_text, "after": after_text,
                "refactoring_type": refactoring["type"], "refactoring_type_id": label_id,
                "repo": repo_name, "commit": sha1, "file": left_loc.get("filePath", ""),
            })

            if max_refactorings and len(pairs) >= max_refactorings:
                logger.info(f"Reached --max-refactorings ({max_refactorings}), stopping early")
                return pairs

    logger.info(f"Built {len(pairs)} before/after pairs, skipped {sum(skipped.values())}")
    for reason, count in skipped.items():
        if count:
            logger.info(f"  skipped ({reason}): {count}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build a before/after code-pairs dataset from mined JSON")
    parser.add_argument("--mined-json", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--repo-name", required=True, help="e.g. 'owner/repo', stored alongside each pair")
    parser.add_argument("--output", default="./refactoring_mining/codegen_pairs.jsonl")
    parser.add_argument("--max-refactorings", type=int, default=None)
    args = parser.parse_args()

    pairs = build_pairs(Path(args.mined_json), Path(args.repo_path), args.repo_name,
                         max_refactorings=args.max_refactorings)
    if not pairs:
        logger.warning("No pairs were produced - nothing appended")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info(f"Appended {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
    main()
