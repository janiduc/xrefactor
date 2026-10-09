"""
The hybrid routing seam: try a deterministic refactoring first, fall back to the
neural generator.

For the patterns implemented here the engine produces an edit it can justify and
the verifier confirms it against javac. When the engine refuses - which it does
often and on purpose - the refusal code is recorded and the neural path takes
over, clearly labelled as unverified. A suggestion is never emitted with a
BROKEN verification, and a refusal is never silently dropped.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import javalang
from loguru import logger

from src.refactor import result as R
from src.refactor.naming import NameProposer, name_is_poor
from src.refactor.patterns.remove_dead_code import remove_dead_code
from src.refactor.patterns.rename_variable import rename_local
from src.refactor.result import RefactorResult
from src.refactor.verify import VerificationResult, verify_refactoring

# Patterns with a deterministic implementation. The rest (move_class,
# split_class, extract_interface, reduce_coupling, consolidate_duplicate_code)
# are multi-file or design-level and stay neural-only.
SUPPORTED_PATTERNS = frozenset({"rename_variable", "improve_naming", "remove_dead_code"})


@dataclass
class EngineOutcome:
    result: RefactorResult
    verification: Optional[VerificationResult] = None

    @property
    def usable(self) -> bool:
        """Applied AND verified. Anything else must not be presented as correct."""
        return self.result.applied and self.verification is not None and self.verification.passed

    def to_dict(self) -> dict:
        payload = {"refactor": self.result.to_dict()}
        if self.verification is not None:
            payload["verification"] = self.verification.to_dict()
        return payload


def _find_method(tree: Any, name: str, line: Optional[int] = None) -> Optional[Any]:
    """Locate a MethodDeclaration by name, preferring the one nearest `line`
    (the CPG keys methods by name alone, so overloads collide)."""
    matches = [node for _path, node in tree.filter(javalang.tree.MethodDeclaration)
               if node.name == name]
    if not matches:
        return None
    if line is None or len(matches) == 1:
        return matches[0]
    return min(matches,
               key=lambda node: abs((node.position[0] if node.position else 0) - line))


def _pick_rename_target(method_node: Any) -> Optional[str]:
    """Choose which local/parameter to rename: the first whose name is actually
    poor. Returns None when every name in the method is already acceptable."""
    for parameter in (method_node.parameters or []):
        if name_is_poor(parameter.name):
            return parameter.name
    for _path, declaration in method_node:
        if isinstance(declaration, javalang.tree.LocalVariableDeclaration):
            for declarator in declaration.declarators:
                if name_is_poor(declarator.name):
                    return declarator.name
    return None


def attempt_refactoring(pattern: str,
                        member_name: str,
                        file_path: Optional[str] = None,
                        source: Optional[str] = None,
                        member_line: Optional[int] = None,
                        proposer: Optional[NameProposer] = None) -> EngineOutcome:
    """Attempt `pattern` on the named member, then compile-verify the result."""
    if pattern not in SUPPORTED_PATTERNS:
        return EngineOutcome(RefactorResult.refuse(
            pattern, R.PATTERN_NOT_SUPPORTED,
            f"'{pattern}' has no deterministic implementation (multi-file or design-level)"))

    if source is None:
        if not file_path or not os.path.exists(file_path):
            return EngineOutcome(RefactorResult.refuse(
                pattern, R.NODE_NOT_FOUND, f"source file not available: {file_path}"))
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                source = handle.read()
        except Exception as e:
            return EngineOutcome(RefactorResult.refuse(
                pattern, R.NODE_NOT_FOUND, f"could not read {file_path}: {e}"))

    try:
        tree = javalang.parse.parse(source)
    except Exception as e:
        return EngineOutcome(RefactorResult.refuse(
            pattern, R.NO_PARSE, f"javalang cannot parse this file: {e}"))

    method_node = _find_method(tree, member_name, member_line)
    if method_node is None:
        return EngineOutcome(RefactorResult.refuse(
            pattern, R.NODE_NOT_FOUND, f"method '{member_name}' not found in the parsed file"))

    if pattern in ("rename_variable", "improve_naming"):
        target = _pick_rename_target(method_node)
        if target is None:
            return EngineOutcome(RefactorResult.refuse(
                pattern, R.NAME_ALREADY_ACCEPTABLE,
                "every local/parameter in this method already has an acceptable name"))
        outcome = rename_local(source, method_node, target, file_tree=tree, proposer=proposer)
    else:  # remove_dead_code
        outcome = remove_dead_code(source, method_node, tree)

    if not outcome.applied:
        return EngineOutcome(outcome)

    basename = os.path.basename(file_path) if file_path else "Subject.java"
    verification = verify_refactoring(source, outcome.modified_source,
                                       file_name=basename,
                                       touched_identifiers=outcome.touched_identifiers)
    if verification.broken:
        # The engine believed the edit was safe and javac disagreed. Never emit
        # it; surface it loudly so the refusal conditions can be tightened.
        logger.warning(f"AST engine produced a BROKEN edit for {pattern} on "
                       f"{member_name}: {verification.verdict} - {verification.detail}")

    return EngineOutcome(outcome, verification)


# Order matters: the least invasive transformation is tried first.
_OPPORTUNITY_ORDER = ("rename_variable", "remove_dead_code")


def find_applicable_pattern(member_name: str,
                            file_path: Optional[str] = None,
                            source: Optional[str] = None,
                            member_line: Optional[int] = None,
                            proposer: Optional[NameProposer] = None
                            ) -> Tuple[Optional[str], Optional[EngineOutcome], List[Tuple[str, str]]]:
    """Ask the engine whether ANY deterministic pattern applies to this member.

    Needed because pattern selection is the real bottleneck, not the engine: the
    classifier predicts `reduce_coupling` for most nodes (it has no
    deterministic implementation) and the detector's poor_naming smells attach to
    variable nodes rather than the enclosing method, so a method sample often
    carries no engine-supported pattern even when a verifiable refactoring
    exists.

    Returns (pattern, outcome, refusals) where `refusals` records what each
    attempted pattern said, so choosing this path is auditable rather than
    silent. Callers MUST record that the applied pattern differs from the
    predicted one.
    """
    refusals: List[Tuple[str, str]] = []
    for pattern in _OPPORTUNITY_ORDER:
        outcome = attempt_refactoring(pattern, member_name, file_path=file_path,
                                       source=source, member_line=member_line,
                                       proposer=proposer)
        if outcome.usable:
            return pattern, outcome, refusals
        refusals.append((pattern, outcome.result.refusal_code
                          or (outcome.verification.verdict if outcome.verification else "UNKNOWN")))
    return None, None, refusals
