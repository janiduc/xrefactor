"""
Shared git plumbing used by build_refactoring_dataset.py and
build_codegen_dataset.py - factored out so both scripts share one
implementation of "resolve a parent commit", "checkout a commit", and
"read a file's content as of a specific commit".
"""

import subprocess
from pathlib import Path
from typing import Optional


def run(cmd, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def git_parent_sha(repo_path: Path, sha1: str) -> Optional[str]:
    result = run(["git", "rev-parse", f"{sha1}^"], repo_path)
    return result.stdout.strip() if result.returncode == 0 else None


def git_checkout(repo_path: Path, sha1: str) -> bool:
    """git checkout exits non-zero if it fails to write ANY file - on Windows this
    routinely happens for a handful of long-named test-fixture files (>260 char
    paths) even when the working tree otherwise switched correctly. Verify by
    checking HEAD actually moved to the target commit, rather than trusting the
    exit code."""
    run(["git", "checkout", "--force", sha1], repo_path)
    head = run(["git", "rev-parse", "HEAD"], repo_path)
    return head.stdout.strip() == sha1


def git_current_ref(repo_path: Path) -> str:
    result = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_path)
    return result.stdout.strip() or "HEAD"


def git_show_file(repo_path: Path, sha1: str, file_path: str) -> Optional[str]:
    """Read a file's content as of a specific commit, without checking out.
    file_path should use forward slashes (git's own convention), matching what
    RefactoringMiner reports. Returns None if the file didn't exist at that
    commit (renamed, deleted, or a bad path)."""
    posix_path = file_path.replace("\\", "/")
    result = run(["git", "show", f"{sha1}:{posix_path}"], repo_path)
    return result.stdout if result.returncode == 0 else None


def slice_lines(content: str, start_line: int, end_line: int) -> Optional[str]:
    """1-indexed, inclusive line range - matches RefactoringMiner's
    startLine/endLine convention. Returns None if the range is out of bounds."""
    lines = content.split("\n")
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        return None
    return "\n".join(lines[start_line - 1:end_line])
