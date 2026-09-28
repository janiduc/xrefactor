"""
Clones a small set of repos (with full git history) and runs RefactoringMiner
against each one's commit history, producing one JSON file per repo.

Requires: git on PATH, a JDK on PATH, and the RefactoringMiner release
extracted under xrefactor/tools/RefactoringMiner-<version>/ (see README.md in
this folder for how it was obtained).

Usage:
    python refactoring_mining/mine_refactorings.py --repos ErikHage/MicroBreweryModel CyiceK/blizzard-game-demo
"""

import argparse
import subprocess
from pathlib import Path

from loguru import logger

THIS_DIR = Path(__file__).parent
DEFAULT_CLONE_DIR = THIS_DIR / "repos"
DEFAULT_OUTPUT_DIR = THIS_DIR / "mined_json"
DEFAULT_RM_HOME = THIS_DIR.parent / "tools" / "RefactoringMiner-3.1.4"


def _run(cmd, cwd=None):
    logger.info(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def clone_repo(org_repo: str, clone_dir: Path) -> Path:
    """Clone <org>/<repo> (full history) if not already present, return local path"""
    local_name = org_repo.replace("/", "_")
    local_path = clone_dir / local_name
    if local_path.exists():
        logger.info(f"{org_repo} already cloned at {local_path}")
        return local_path
    
    clone_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{org_repo}.git"
    result = _run(["git", "clone", url, str(local_path)])
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed for {org_repo}: {result.stderr}")
    return local_path


def default_branch(repo_path: Path) -> str:
    result = _run(["git", "branch", "--show-current"], cwd=str(repo_path))
    branch = result.stdout.strip()
    if not branch:
        # Detached HEAD fallback
        result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_path))
        branch = result.stdout.strip()
    return branch or "main"


def run_refactoring_miner(repo_path: Path, branch: str, rm_home: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java", "-cp", f"{rm_home}\\lib\\*", "org.refactoringminer.RefactoringMiner",
        "-a", str(repo_path), branch, "-json", str(output_path)
    ]
    logger.info(f"Mining {repo_path.name} (branch={branch})...")
    result = _run(cmd)
    logger.info(result.stdout[-2000:] if result.stdout else "(no stdout)")
    if result.stderr:
        logger.info(result.stderr[-2000:])
    if not output_path.exists():
        raise RuntimeError(f"RefactoringMiner did not produce output for {repo_path.name}")
    logger.info(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mine real refactorings from repo git history")
    parser.add_argument("--repos", nargs="+", required=True, help="One or more 'org/repo' GitHub slugs")
    parser.add_argument("--clone-dir", default=str(DEFAULT_CLONE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rm-home", default=str(DEFAULT_RM_HOME))
    args = parser.parse_args()
    
    clone_dir = Path(args.clone_dir)
    output_dir = Path(args.output_dir)
    rm_home = Path(args.rm_home)
    
    for org_repo in args.repos:
        try:
            repo_path = clone_repo(org_repo, clone_dir)
            branch = default_branch(repo_path)
            output_path = output_dir / f"{org_repo.replace('/', '_')}.json"
            run_refactoring_miner(repo_path, branch, rm_home, output_path)
        except Exception as e:
            logger.error(f"Failed to mine {org_repo}: {e}")


if __name__ == "__main__":
    main()
