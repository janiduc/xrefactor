"""
Structural correctness checks: does GENERATED code actually look like a
correct instance of the specific refactoring pattern it was supposed to
apply, or just plausible-looking text? This is the final link in the
smell -> pattern -> generated-code chain: Phase 4.5's smell_detector
identifies a smell and maps it to a pattern, Stage 3 generates code for
that pattern, and these validators check the generated code structurally
conforms to it.

Every validator here is a STRUCTURAL heuristic, not a semantic-equivalence
proof - none of them execute or formally verify behavior preservation.
Where a check is especially approximate, its docstring says so explicitly.
They operate on the SAME before/after convention build_codegen_dataset.py
mined training pairs with: for most patterns, "after" is the modified
version of the same code element; for extract_method specifically,
"after" is RefactoringMiner's newly-created sibling method, not the
shortened original with a call-site (a documented simplification carried
through consistently from mining to validation).

Usage:
    from src.transformer.pattern_validators import validate
    result = validate("extract_method", before_text, after_text)
    # {"passed": bool, "reason": str, ...pattern-specific evidence}
"""

import re
from typing import Any, Dict, List, Optional

import javalang
from javalang.tokenizer import Identifier

from src.refactor.offsets import tokenize_cached


def _try_parse_member(snippet: str):
    """Best-effort parse a bare method/field/class snippet by wrapping it in a
    dummy compilation unit - javalang.parse.parse requires a full unit, but
    mined before/after snippets are just one member's source text."""
    wrapped = f"class __Dummy__ {{ {snippet}\n}}"
    try:
        return javalang.parse.parse(wrapped)
    except Exception:
        try:
            # Class/interface-level snippets are already a type declaration.
            return javalang.parse.parse(snippet)
        except Exception:
            return None


def _method_declarations(tree) -> List[Any]:
    if tree is None:
        return []
    return [node for _, node in tree.filter(javalang.tree.MethodDeclaration)]


def _type_declarations(tree) -> List[Any]:
    """javalang's tree.filter() does not accept a tuple of types the way
    isinstance() does (it silently matches nothing), so each type is
    filtered separately and combined. Excludes the synthetic __Dummy__
    wrapper class _try_parse_member introduces - without this, ANY
    successfully-parsed snippet (even a bare method) would falsely count
    as "containing a type declaration", since the wrapper itself is one."""
    if tree is None:
        return []
    classes = [node for _, node in tree.filter(javalang.tree.ClassDeclaration) if node.name != "__Dummy__"]
    interfaces = [node for _, node in tree.filter(javalang.tree.InterfaceDeclaration) if node.name != "__Dummy__"]
    return classes + interfaces


def _identifiers(snippet: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", snippet)


def _boolean_and_branch_op_count(snippet: str) -> int:
    return snippet.count("&&") + snippet.count("||")


def _non_whitespace_len(snippet: str) -> int:
    return len(re.sub(r"\s+", "", snippet))


def _non_identifier_skeleton(snippet: str) -> Optional[List[str]]:
    """Token values with every Identifier removed.

    Two snippets that differ only by renaming share this skeleton exactly, so
    comparing it is a precise test for "nothing but names changed".
    """
    tokens = tokenize_cached(snippet)
    if tokens is None:
        return None
    return [t.value for t in tokens if not isinstance(t, Identifier)]


def _try_parse_statements(snippet: str):
    """Parse a snippet as a sequence of statements inside a method body."""
    wrapped = "class __Dummy__ { void __m__() {\n" + snippet + "\n} }"
    try:
        return javalang.parse.parse(wrapped)
    except Exception:
        return None


def parses_as_member(snippet: str) -> bool:
    """Is the snippet syntactically valid Java of SOME kind - a member/type
    declaration, or a sequence of statements?

    Both forms are accepted because mined before/after pairs legitimately
    contain either (a whole method, or statements lifted out of one). This is
    the gate that stops meaningless text being scored at all: without it,
    `validate_reduce_coupling` returned passed=True for
    'public boolean is  private final Collection<>;' purely because its
    qualified-call count went 0 -> 0.
    """
    if not snippet or not snippet.strip():
        return False
    if _try_parse_member(snippet) is not None:
        return True
    return _try_parse_statements(snippet) is not None


def _is_degenerate(snippet: str) -> bool:
    """Catches the specific failure mode an undertrained/greedy-decoded model
    produces: short-phrase repetition ("private final Collection" over and
    over) that individual metrics like "qualified-call count" or "shrink
    ratio" can satisfy without the text being meaningful code at all. Flags
    text where token-level repetition dominates."""
    tokens = _identifiers(snippet)
    if len(tokens) < 4:
        # Previously this short-circuited to "not degenerate", which let short
        # junk like 'AnalysisState<>;' through. Short text is only acceptable if
        # it actually parses as a declaration.
        return not parses_as_member(snippet)
    unique_ratio = len(set(tokens)) / len(tokens)
    return unique_ratio < 0.35


def validate_extract_method(before: str, after: str) -> Dict[str, Any]:
    """Checks 'after' parses as a valid, distinct method declaration, and is
    shorter than 'before' - consistent with our mining convention where
    'after' is the newly-extracted sibling method, not the shortened
    original-plus-call-site. Does NOT verify the extracted method is
    actually called anywhere (that would require the modified original,
    which this convention doesn't capture)."""
    after_tree = _try_parse_member(after)
    after_methods = _method_declarations(after_tree)
    if not after_methods:
        return {"passed": False, "reason": "after does not parse as a method declaration"}
    if after.strip() == before.strip():
        return {"passed": False, "reason": "after is identical to before"}
    shorter = _non_whitespace_len(after) < _non_whitespace_len(before)
    return {"passed": shorter, "reason": "after is a valid, shorter, distinct method" if shorter
            else "after did not get shorter than before", "after_len": _non_whitespace_len(after),
            "before_len": _non_whitespace_len(before)}


def validate_rename_variable(before: str, after: str) -> Dict[str, Any]:
    """A rename changes identifiers and NOTHING else.

    The strong check is structural: with every Identifier token removed, the
    remaining token sequence (keywords, separators, operators, literals) must be
    IDENTICAL before and after. That is what distinguishes a rename from a
    rewrite that happens to share some names, and it is far harder to satisfy by
    accident than the previous "small swapped identifier set" heuristic.
    """
    before_tokens = tokenize_cached(before if before.strip().endswith(("}", ";"))
                                    else before)
    after_tokens = tokenize_cached(after)

    before_skeleton = _non_identifier_skeleton(before)
    after_skeleton = _non_identifier_skeleton(after)
    if before_skeleton is None or after_skeleton is None:
        return {"passed": False, "reason": "could not tokenize both sides"}
    if before_skeleton != after_skeleton:
        return {"passed": False,
                "reason": "non-identifier token sequence changed, so this is a rewrite not a rename"}

    before_ids, after_ids = _identifiers(before), _identifiers(after)
    if len(before_ids) != len(after_ids):
        return {"passed": False, "reason": "identifier count changed"}

    changed_positions = [(b, a) for b, a in zip(before_ids, after_ids) if b != a]
    if not changed_positions:
        return {"passed": False, "reason": "nothing was renamed"}

    distinct_before = {b for b, _ in changed_positions}
    if len(distinct_before) != 1:
        return {"passed": False,
                "reason": f"{len(distinct_before)} distinct identifiers changed; a rename changes exactly one"}
    distinct_after = {a for _, a in changed_positions}
    if len(distinct_after) != 1:
        return {"passed": False, "reason": "one name was replaced by several different names"}

    return {"passed": True,
            "reason": f"exactly one identifier renamed ({next(iter(distinct_before))} -> "
                      f"{next(iter(distinct_after))}) with all other tokens unchanged",
            "renamed_from": next(iter(distinct_before)),
            "renamed_to": next(iter(distinct_after)),
            # Number of token POSITIONS rewritten, not distinct name pairs.
            "occurrences": len(changed_positions)}


def validate_remove_dead_code(before: str, after: str) -> Dict[str, Any]:
    """Approximate: checks 'after' is meaningfully shorter than 'before'
    (something was removed) - cannot verify the REMOVED part was actually
    the unreachable part, only that a removal-shaped change happened."""
    before_len, after_len = _non_whitespace_len(before), _non_whitespace_len(after)
    if before_len == 0:
        return {"passed": False, "reason": "empty before text"}
    shrink_ratio = 1 - (after_len / before_len)
    passed = shrink_ratio > 0.15
    return {"passed": passed, "reason": f"after shrank by {shrink_ratio*100:.0f}%" if passed
            else "after did not shrink meaningfully", "shrink_ratio": round(shrink_ratio, 3)}


def validate_consolidate_duplicate_code(before: str, after: str) -> Dict[str, Any]:
    """Checks 'after' parses as a valid, distinct method declaration - the
    consolidated/shared method the model should produce. Cannot verify it
    is actually shared by the ORIGINAL duplicate pair without the sibling
    method, which this validator's (before, after) signature doesn't carry."""
    after_tree = _try_parse_member(after)
    after_methods = _method_declarations(after_tree)
    passed = bool(after_methods) and after.strip() != before.strip()
    return {"passed": passed, "reason": "after is a valid, distinct method" if passed
            else "after is not a valid distinct method declaration"}


def validate_simplify_condition(before: str, after: str) -> Dict[str, Any]:
    """Checks the count of boolean operators (&&/||) strictly decreases.
    Purely structural - does NOT verify the simplified condition is
    logically equivalent to the original (that needs an actual equivalence
    checker, out of scope)."""
    before_ops, after_ops = _boolean_and_branch_op_count(before), _boolean_and_branch_op_count(after)
    passed = after_ops < before_ops
    return {"passed": passed, "reason": f"boolean operators {before_ops} -> {after_ops}",
            "before_ops": before_ops, "after_ops": after_ops}


def validate_split_class(before: str, after: str) -> Dict[str, Any]:
    """Checks 'after' contains a class/interface declaration distinct from
    'before' - approximates "a new class/interface appeared" without
    verifying the original class's member count actually decreased
    correspondingly (would need the modified original class, not captured
    by this mining convention)."""
    after_tree = _try_parse_member(after)
    types_found = _type_declarations(after_tree)
    passed = bool(types_found) and after.strip() != before.strip()
    return {"passed": passed, "reason": "after contains a distinct type declaration" if passed
            else "after does not contain a distinct type declaration"}


def validate_extract_interface(before: str, after: str) -> Dict[str, Any]:
    """Checks 'after' specifically parses as an INTERFACE declaration."""
    after_tree = _try_parse_member(after)
    interfaces = [n for _, n in (after_tree.filter(javalang.tree.InterfaceDeclaration) if after_tree else [])]
    passed = bool(interfaces)
    return {"passed": passed, "reason": "after is a valid interface declaration" if passed
            else "after does not parse as an interface declaration"}


def validate_reduce_coupling(before: str, after: str) -> Dict[str, Any]:
    """Approximate: counts qualified-call-like patterns (word.word() ) as a
    proxy for external dependencies, and checks the count doesn't increase.
    A real coupling reduction should show this metric decrease or hold
    steady, not go up."""
    pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\s*\(")
    before_calls, after_calls = len(pattern.findall(before)), len(pattern.findall(after))
    passed = after_calls <= before_calls
    return {"passed": passed, "reason": f"qualified calls {before_calls} -> {after_calls}",
            "before_qualified_calls": before_calls, "after_qualified_calls": after_calls}


def validate_move_class(before: str, after: str) -> Dict[str, Any]:
    """Checks 'after' parses as a valid member/type declaration, distinct
    from 'before'. Cannot verify the class actually moved to a different
    package/file from text alone - that requires file-path metadata this
    validator's signature doesn't carry (see pipeline-level checks, which
    do have file paths, for a stronger version of this check)."""
    after_tree = _try_parse_member(after)
    valid = bool(_method_declarations(after_tree) or _type_declarations(after_tree))
    passed = valid and after.strip() != before.strip()
    return {"passed": passed, "reason": "after is a valid, distinct declaration" if passed
            else "after is not a valid distinct declaration"}


def validate_improve_naming(before: str, after: str) -> Dict[str, Any]:
    """Same shape as rename_variable but framed for method/class-level
    naming - a small, swapped identifier set, not a wholesale rewrite."""
    return validate_rename_variable(before, after)


_VALIDATORS = {
    "extract_method": validate_extract_method,
    "move_class": validate_move_class,
    "rename_variable": validate_rename_variable,
    "consolidate_duplicate_code": validate_consolidate_duplicate_code,
    "remove_dead_code": validate_remove_dead_code,
    "simplify_condition": validate_simplify_condition,
    "split_class": validate_split_class,
    "extract_interface": validate_extract_interface,
    "reduce_coupling": validate_reduce_coupling,
    "improve_naming": validate_improve_naming,
}


def validate(refactoring_type: str, before: str, after: str) -> Dict[str, Any]:
    """Dispatch to the validator for `refactoring_type` (one of the 10
    pattern names in src.gnn.refactoring_types.REFACTORING_TYPE_TO_ID).

    Two gates run BEFORE any pattern-specific metric, because the
    pattern-specific metrics are individually satisfiable by text that is not
    code at all:
      * a degeneracy check (repetition), and
      * a mandatory PARSE gate.

    Without the parse gate, `validate_reduce_coupling` returned passed=True for
    'public boolean is  private final Collection<>;' simply because its
    qualified-call count went 0 -> 0. Any metric comparison on unparseable text
    is meaningless, so it is no longer reached.
    """
    validator = _VALIDATORS.get(refactoring_type)
    if validator is None:
        return {"passed": False, "reason": f"unknown pattern: {refactoring_type}"}

    # Deletion is the refactoring for remove_dead_code, so an empty result is
    # legitimate there and only there.
    if not after or not after.strip():
        if refactoring_type == "remove_dead_code":
            removed = _non_whitespace_len(before) > 0
            return {"passed": removed,
                    "reason": "member removed entirely" if removed
                              else "nothing was removed (before was empty too)",
                    "removed_entirely": True}
        return {"passed": False, "reason": "after is empty"}

    if _is_degenerate(after):
        return {"passed": False, "reason": "after is degenerate/repetitive text, not meaningful code"}

    if not parses_as_member(after):
        return {"passed": False, "reason": "after does not parse as a Java declaration"}

    try:
        return validator(before, after)
    except Exception as e:
        return {"passed": False, "reason": f"validator raised: {e}"}
