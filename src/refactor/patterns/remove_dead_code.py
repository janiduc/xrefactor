"""
remove_dead_code: delete a provably-unused private method, or statements that
cannot be reached.

The CPG's call graph only records same-repo invocations, so the smell detector
flags anything it sees no caller for - which on a Spring/Android/JUnit codebase
means most of the file (measured: 545 of 619 findings on one held-out repo were
dead_code). Deleting on that signal alone would be destructive.

The gate that fixes this is simple and strict: only `private` methods are ever
deleted. A framework-invoked method (@RequestMapping handler, @Bean factory,
JUnit @Test, Android lifecycle override) is never private, so it is structurally
out of reach of this refactoring. On top of that: no annotations at all, no
in-file references, and not one of the serialization/reflection hooks the JVM
calls by name even when private.
"""

import re
from typing import Any, List, Optional, Set

import javalang

from src.refactor import result as R
from src.refactor.extent import member_extent, top_level_statement_extents
from src.refactor.result import RefactorResult

PATTERN = "remove_dead_code"

# Called by name via serialization/reflection even when private, so "no caller
# in this file" says nothing about whether they are needed.
_REFLECTION_HOOKS = {
    "readObject", "writeObject", "readObjectNoData", "readResolve", "writeReplace",
    "finalize", "valueOf", "main", "values", "clone",
}

# Statements after one of these in the same block cannot be reached.
_TERMINAL_STATEMENTS = (
    javalang.tree.ReturnStatement,
    javalang.tree.ThrowStatement,
    javalang.tree.BreakStatement,
    javalang.tree.ContinueStatement,
)


def _method_is_referenced(file_tree: Any, method_name: str, method_node: Any) -> bool:
    """Any invocation or method reference naming this method, anywhere in the file."""
    for _path, node in file_tree.filter(javalang.tree.MethodInvocation):
        if node.member == method_name:
            return True
    for _path, node in file_tree.filter(javalang.tree.MethodReference):
        if getattr(node.method, "member", None) == method_name or \
                getattr(node, "method", None) == method_name:
            return True
    # A string mentioning the name could be reflective use; be conservative.
    for _path, node in file_tree.filter(javalang.tree.Literal):
        value = node.value or ""
        if value.startswith('"') and method_name in value:
            return True
    return False


def _leading_javadoc_start(source: str, member_start: int) -> int:
    """Offset of a javadoc block immediately preceding the member, else
    `member_start`. Deleting a method without its javadoc would leave an orphan
    comment describing nothing."""
    preceding = source[:member_start]
    stripped = preceding.rstrip()
    if not stripped.endswith("*/"):
        return member_start
    start = stripped.rfind("/**")
    if start == -1:
        start = stripped.rfind("/*")
    if start == -1:
        return member_start
    # Only absorb the comment if nothing but whitespace separates it.
    between = source[stripped.rfind("*/") + 2:member_start]
    if between.strip():
        return member_start
    return start


def remove_dead_method(source: str,
                       method_node: Any,
                       file_tree: Any) -> RefactorResult:
    """Delete `method_node` if it is provably unused within this file."""
    modifiers = set(getattr(method_node, "modifiers", None) or ())

    if "private" not in modifiers:
        return RefactorResult.refuse(
            PATTERN, R.NOT_PRIVATE,
            "only private methods are deletable: anything else may be called from "
            "another file or invoked by a framework")

    if getattr(method_node, "annotations", None):
        names = [a.name for a in method_node.annotations if getattr(a, "name", None)]
        return RefactorResult.refuse(PATTERN, R.HAS_ANNOTATIONS,
                                      f"carries annotations {names} which may drive reflective use")

    if method_node.name in _REFLECTION_HOOKS:
        return RefactorResult.refuse(PATTERN, R.REFLECTION_HOOK,
                                      f"'{method_node.name}' is invoked by name by the JVM/serialization")

    if _method_is_referenced(file_tree, method_node.name, method_node):
        return RefactorResult.refuse(PATTERN, R.STILL_REFERENCED,
                                      f"'{method_node.name}' is still referenced in this file")

    extent = member_extent(source, method_node)
    if extent is None:
        return RefactorResult.refuse(PATTERN, R.NO_EXTENT, "could not determine the method extent")

    start = _leading_javadoc_start(source, extent.start_offset)
    end = extent.end_offset

    # Absorb the whitespace/newline the member occupied so no blank gap is left.
    line_start = source.rfind("\n", 0, start) + 1
    if source[line_start:start].strip() == "":
        start = line_start
    while end < len(source) and source[end] in " \t":
        end += 1
    if end < len(source) and source[end] == "\n":
        end += 1

    modified = source[:start] + source[end:]

    return RefactorResult(
        applied=True,
        pattern=PATTERN,
        modified_source=modified,
        original_member=extent.text(source),
        refactored_member="",  # the member is gone
        touched_identifiers=[method_node.name],
        metadata={
            "removed": "method",
            "method_name": method_node.name,
            "removed_lines": extent.end_line - extent.start_line + 1,
        },
    )


def remove_unreachable_statements(source: str, method_node: Any) -> RefactorResult:
    """Delete top-level statements that follow a statement which cannot complete
    normally (return/throw/break/continue) in the same block."""
    if not method_node.body:
        return RefactorResult.refuse(PATTERN, R.NO_BODY, "method has no body")

    extents = top_level_statement_extents(source, method_node)
    if extents is None:
        return RefactorResult.refuse(PATTERN, R.NO_EXTENT, "could not place the method's statements")

    terminal_index: Optional[int] = None
    for index, statement in enumerate(method_node.body):
        if isinstance(statement, _TERMINAL_STATEMENTS):
            terminal_index = index
            break

    if terminal_index is None or terminal_index >= len(method_node.body) - 1:
        return RefactorResult.refuse(PATTERN, R.NOTHING_UNREACHABLE,
                                      "no statements follow a terminal statement")

    # Everything after the terminal statement is unreachable.
    first_dead = extents[terminal_index + 1]
    last_dead = extents[-1]
    start, end = first_dead.start_offset, last_dead.end_offset

    line_start = source.rfind("\n", 0, start) + 1
    if source[line_start:start].strip() == "":
        start = line_start

    modified = source[:start] + source[end:]

    member = member_extent(source, method_node)
    return RefactorResult(
        applied=True,
        pattern=PATTERN,
        modified_source=modified,
        original_member=member.text(source) if member else None,
        refactored_member=None,
        touched_identifiers=[],
        metadata={
            "removed": "unreachable_statements",
            "statements_removed": len(method_node.body) - terminal_index - 1,
            "after_statement": type(method_node.body[terminal_index]).__name__,
        },
    )


def remove_dead_code(source: str, method_node: Any, file_tree: Any) -> RefactorResult:
    """Try the narrow, local fix first (unreachable statements), then whole-method
    deletion. Returns the first that applies, else the method-level refusal,
    which is the more informative one."""
    unreachable = remove_unreachable_statements(source, method_node)
    if unreachable.applied:
        return unreachable
    return remove_dead_method(source, method_node, file_tree)
