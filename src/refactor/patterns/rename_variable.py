"""
rename_variable / improve_naming: rename a local variable or parameter.

Tokens carry no binding information, so "rewrite every Identifier token matching
the old name" is wrong whenever the name is shadowed, is really a field, is a
method name, is a label, or appears as a qualified member (`other.x`). The safe
construction is therefore three layers:

  1. an AST gate that PROVES the name is a single local/parameter of this method
     (R1-R7 below), refusing otherwise;
  2. token rules that exclude occurrences which are not references to it;
  3. a cross-check that the number of tokens about to be rewritten equals
     1 (the declaration) + the number of AST reference nodes. Any disagreement
     means one layer saw something the other did not, so we refuse.

Only locals and parameters are in scope. Renaming a field or a method would
require editing every file that references it, which cannot be verified without
whole-program resolution - those refuse with FIELD_RENAME_CROSS_FILE /
METHOD_RENAME_CROSS_FILE.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import javalang
from javalang.tokenizer import Identifier, Keyword, Separator

from src.refactor import result as R
from src.refactor.extent import member_extent
from src.refactor.naming import NameProposer, RenameContext, choose_name, name_is_poor
from src.refactor.offsets import pos_to_offset, tokenize_cached
from src.refactor.result import RefactorResult

PATTERN = "rename_variable"

# A parameter whose NAME is load-bearing at runtime via reflection must not be
# renamed: these frameworks bind by parameter name when no explicit value is given.
_REFLECTIVE_PARAM_ANNOTATIONS = {
    "PathVariable", "RequestParam", "RequestHeader", "CookieValue", "MatrixVariable",
    "ModelAttribute", "SessionAttribute", "PathParam", "QueryParam", "FormParam",
    "HeaderParam", "Param", "Value", "Named", "Qualifier", "JsonProperty",
    "ConstructorProperties",
}

# An Identifier preceded by one of these is not a reference to a local.
_SKIP_IF_PREV_VALUE = {".", "::", "@"}
_SKIP_IF_PREV_KEYWORD = {"new", "class", "interface", "enum", "extends", "implements",
                         "throws", "instanceof", "case"}


def _declarations_of(method_node, name: str) -> List[Tuple[Any, Tuple]]:
    """Every declaration of `name` inside the method, with its ancestor path."""
    found: List[Tuple[Any, Tuple]] = []

    for parameter in (method_node.parameters or []):
        if parameter.name == name:
            found.append((parameter, ()))

    for path, node in method_node:
        if isinstance(node, javalang.tree.LocalVariableDeclaration):
            for declarator in node.declarators:
                if declarator.name == name:
                    found.append((node, path))
        elif isinstance(node, javalang.tree.VariableDeclaration):
            for declarator in (node.declarators or []):
                if declarator.name == name:
                    found.append((node, path))
        elif isinstance(node, javalang.tree.TryResource):
            if node.name == name:
                found.append((node, path))
        elif isinstance(node, javalang.tree.CatchClauseParameter):
            if node.name == name:
                found.append((node, path))
        elif isinstance(node, javalang.tree.LambdaExpression):
            for parameter in (node.parameters or []):
                param_name = getattr(parameter, "name", None) or getattr(parameter, "member", None)
                if param_name == name:
                    found.append((parameter, path))
    return found


def _has_nested_scope(method_node) -> bool:
    """Does the method contain a lambda, anonymous class body or local class?"""
    for _path, node in method_node:
        if isinstance(node, javalang.tree.LambdaExpression):
            return True
        if isinstance(node, javalang.tree.ClassCreator) and node.body:
            return True
        if isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
            return True
    return False


def _reference_nodes(method_node, name: str) -> int:
    """Count AST nodes that reference the local `name`.

    MemberReferences that are selectors of `this`/`super` are excluded: those
    denote the FIELD of that name, and the token layer skips them too (their
    previous token is `.`), so excluding them keeps the two layers consistent.
    """
    qualified_selector_ids: Set[int] = set()
    for _path, node in method_node:
        if isinstance(node, (javalang.tree.This, javalang.tree.SuperMemberReference)):
            for selector in (getattr(node, "selectors", None) or []):
                qualified_selector_ids.add(id(selector))

    count = 0
    for _path, node in method_node:
        if isinstance(node, javalang.tree.MemberReference):
            if id(node) in qualified_selector_ids:
                continue
            qualifier = node.qualifier or ""
            if not qualifier and node.member == name:
                count += 1
            elif qualifier.split(".")[0] == name:
                count += 1
        elif isinstance(node, javalang.tree.MethodInvocation):
            qualifier = node.qualifier or ""
            if qualifier and qualifier.split(".")[0] == name:
                count += 1
    return count


def _labels_in(method_node) -> Set[str]:
    labels = set()
    for _path, node in method_node:
        label = getattr(node, "label", None)
        if label:
            labels.add(label)
    return labels


def _type_names_in(method_node) -> Set[str]:
    names = set()
    for _path, node in method_node:
        if isinstance(node, javalang.tree.ReferenceType) and node.name:
            names.add(node.name)
    for type_parameter in (getattr(method_node, "type_parameters", None) or []):
        if getattr(type_parameter, "name", None):
            names.add(type_parameter.name)
    return names


def _annotation_names(node) -> Set[str]:
    names = set()
    for annotation in (getattr(node, "annotations", None) or []):
        if getattr(annotation, "name", None):
            names.add(annotation.name.split(".")[-1])
    return names


def _has_explicit_annotation_value(node, wanted: Set[str]) -> bool:
    """True when a reflective annotation supplies an explicit name, which makes
    renaming the parameter safe again."""
    for annotation in (getattr(node, "annotations", None) or []):
        simple = (getattr(annotation, "name", "") or "").split(".")[-1]
        if simple in wanted:
            if getattr(annotation, "element", None):
                return True
    return False


def _declared_type_text(source: str, declaration: Any) -> str:
    """Type text spliced verbatim from the source, so generics and array
    dimensions come through exactly rather than being reconstructed."""
    type_node = getattr(declaration, "type", None)
    if type_node is None or getattr(type_node, "position", None) is None:
        return ""
    tokens = tokenize_cached(source)
    if tokens is None:
        return ""
    start_line, start_col = type_node.position
    start = pos_to_offset(source, start_line, start_col)

    # Read up to the declared name.
    name = getattr(declaration, "name", None)
    if name is None:
        declarators = getattr(declaration, "declarators", None) or []
        name = declarators[0].name if declarators else None
    if name is None:
        return ""
    for token in tokens:
        token_offset = pos_to_offset(source, token.position[0], token.position[1])
        if token_offset > start and isinstance(token, Identifier) and token.value == name:
            return source[start:token_offset].strip()
    return ""


def _initializer_source(source: str, declaration: Any, name: str) -> str:
    for declarator in (getattr(declaration, "declarators", None) or []):
        if declarator.name == name and declarator.initializer is not None:
            position = getattr(declarator.initializer, "position", None)
            if position is None:
                return ""
            start = pos_to_offset(source, position[0], position[1])
            end = source.find(";", start)
            return source[start:end if end != -1 else start + 80]
    return ""


def rename_local(source: str,
                 method_node: Any,
                 old_name: str,
                 file_tree: Any = None,
                 new_name: Optional[str] = None,
                 proposer: Optional[NameProposer] = None,
                 enforce_poor_name: bool = True) -> RefactorResult:
    """Rename a local variable or parameter of `method_node` across the method.

    Returns an applied result carrying the whole modified FILE (for compile
    verification) and the refactored member text, or a refusal with a code.
    """
    extent = member_extent(source, method_node)
    if extent is None:
        return RefactorResult.refuse(PATTERN, R.NO_EXTENT,
                                      "could not determine the method's source extent")

    tokens = tokenize_cached(source)
    if tokens is None:
        return RefactorResult.refuse(PATTERN, R.NO_PARSE, "source does not lex")

    # ---- Precision gate: do not churn names that are already fine. --------- #
    if enforce_poor_name and new_name is None and not name_is_poor(old_name):
        return RefactorResult.refuse(PATTERN, R.NAME_ALREADY_ACCEPTABLE,
                                      f"'{old_name}' is already an acceptable name")

    # ---- R1: exactly one declaration of the name in this method ----------- #
    declarations = _declarations_of(method_node, old_name)
    if len(declarations) == 0:
        return RefactorResult.refuse(PATTERN, R.TARGET_NOT_A_LOCAL,
                                      f"'{old_name}' is not declared as a local or parameter here")
    if len(declarations) > 1:
        return RefactorResult.refuse(PATTERN, R.SHADOWED_OR_REDECLARED,
                                      f"'{old_name}' is declared {len(declarations)} times in this method")
    declaration, declaration_path = declarations[0]

    # ---- R2: the declaration must not sit in a nested scope --------------- #
    for ancestor in declaration_path:
        if isinstance(ancestor, (javalang.tree.LambdaExpression,
                                 javalang.tree.ClassDeclaration,
                                 javalang.tree.InterfaceDeclaration)):
            return RefactorResult.refuse(PATTERN, R.NESTED_SCOPE_DECLARATION,
                                          "declared inside a lambda or local/anonymous class")
        if isinstance(ancestor, javalang.tree.ClassCreator) and ancestor.body:
            return RefactorResult.refuse(PATTERN, R.NESTED_SCOPE_DECLARATION,
                                          "declared inside an anonymous class body")

    # ---- R3: same-named field plus a capturing scope is unresolvable ------ #
    field_names: Set[str] = set()
    method_names: Set[str] = set()
    type_names: Set[str] = set()
    if file_tree is not None:
        for _path, node in file_tree.filter(javalang.tree.FieldDeclaration):
            for declarator in node.declarators:
                field_names.add(declarator.name)
        for _path, node in file_tree.filter(javalang.tree.MethodDeclaration):
            method_names.add(node.name)
        for _path, node in file_tree.filter(javalang.tree.ClassDeclaration):
            type_names.add(node.name)
        for _path, node in file_tree.filter(javalang.tree.InterfaceDeclaration):
            type_names.add(node.name)

    if old_name in field_names and _has_nested_scope(method_node):
        return RefactorResult.refuse(
            PATTERN, R.AMBIGUOUS_FIELD_CAPTURE,
            f"'{old_name}' is also a field and this method captures scope (lambda/anon class)")

    # ---- R4: the name must not be a label --------------------------------- #
    if old_name in _labels_in(method_node):
        return RefactorResult.refuse(PATTERN, R.NAME_IS_LABEL, f"'{old_name}' is used as a label")

    # ---- R5/R6: must not collide with a member or a type ------------------ #
    if old_name == method_node.name or old_name in method_names or old_name in type_names:
        return RefactorResult.refuse(PATTERN, R.NAME_COLLIDES_WITH_MEMBER,
                                      f"'{old_name}' also names a method or type in this file")
    if old_name in _type_names_in(method_node):
        return RefactorResult.refuse(PATTERN, R.NAME_IS_TYPE, f"'{old_name}' is used as a type name")

    # ---- R7: reflective parameter names ----------------------------------- #
    is_parameter = isinstance(declaration, javalang.tree.FormalParameter)
    if is_parameter:
        annotations = _annotation_names(declaration) | _annotation_names(method_node)
        reflective = annotations & _REFLECTIVE_PARAM_ANNOTATIONS
        if reflective and not _has_explicit_annotation_value(declaration, reflective):
            return RefactorResult.refuse(
                PATTERN, R.REFLECTIVE_PARAM_NAME,
                f"parameter name is bound reflectively by @{sorted(reflective)[0]}")

    # ---- Choose the replacement name -------------------------------------- #
    name_source = "explicit"
    rejected: List[dict] = []
    if new_name is None:
        context = RenameContext(
            old_name=old_name,
            kind="parameter" if is_parameter else "local",
            declared_type_text=_declared_type_text(source, declaration),
            initializer_source=_initializer_source(source, declaration, old_name),
            method_source=extent.text(source),
            enclosing_class_name=next(iter(type_names), ""),
        )
        new_name, name_source, rejected = choose_name(context, source, proposer)
        if new_name is None:
            return RefactorResult.refuse(PATTERN, R.NO_VALID_NEW_NAME,
                                          "no proposed name passed validation")

    # ---- Token selection -------------------------------------------------- #
    selected: List[Tuple[int, int]] = []  # (offset, length)
    for index, token in enumerate(tokens):
        if not isinstance(token, Identifier) or token.value != old_name:
            continue
        offset = pos_to_offset(source, token.position[0], token.position[1])
        if offset < extent.start_offset or offset >= extent.end_offset:
            continue
        previous = tokens[index - 1] if index > 0 else None
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if previous is not None:
            if previous.value in _SKIP_IF_PREV_VALUE:
                continue
            if isinstance(previous, Keyword) and previous.value in _SKIP_IF_PREV_KEYWORD:
                continue
        # A following '(' means this is a method name, not a variable.
        if following is not None and isinstance(following, Separator) and following.value == "(":
            continue
        selected.append((offset, len(token.value)))

    # ---- Cross-check the two layers --------------------------------------- #
    expected = 1 + _reference_nodes(method_node, old_name)
    if len(selected) != expected:
        return RefactorResult.refuse(
            PATTERN, R.TOKEN_AST_MISMATCH,
            f"token layer found {len(selected)} occurrence(s) but the AST implies {expected}")
    if not selected:
        return RefactorResult.refuse(PATTERN, R.TOKEN_AST_MISMATCH, "no occurrences selected")

    # ---- Apply splices, highest offset first ------------------------------ #
    modified = source
    for offset, length in sorted(selected, reverse=True):
        modified = modified[:offset] + new_name + modified[offset + length:]

    javadoc_updated = False
    if is_parameter:
        modified, javadoc_updated = _update_javadoc_param(modified, extent.start_offset,
                                                           old_name, new_name, len(selected),
                                                           len(new_name) - len(old_name))

    new_extent = None
    try:
        new_tree = javalang.parse.parse(modified)
        for _path, node in new_tree.filter(javalang.tree.MethodDeclaration):
            if node.name == method_node.name:
                new_extent = member_extent(modified, node)
                break
    except Exception:
        new_extent = None

    return RefactorResult(
        applied=True,
        pattern=PATTERN,
        modified_source=modified,
        original_member=extent.text(source),
        refactored_member=new_extent.text(modified) if new_extent else None,
        touched_identifiers=[old_name, new_name],
        metadata={
            "old_name": old_name,
            "new_name": new_name,
            "name_source": name_source,
            "rejected_candidates": rejected,
            "occurrences_renamed": len(selected),
            "javadoc_param_updated": javadoc_updated,
            "target_kind": "parameter" if is_parameter else "local",
        },
    )


def _update_javadoc_param(source: str,
                          member_start: int,
                          old_name: str,
                          new_name: str,
                          splice_count: int,
                          per_splice_delta: int) -> Tuple[str, bool]:
    """Rewrite `@param <old>` in the javadoc block immediately above the member.

    Identifiers in comments produce no tokens, so a javadoc tag would otherwise
    go stale and silently disagree with the code.
    """
    # Member offsets shifted by the splices applied before this point.
    shifted_start = member_start
    window_start = max(0, shifted_start - 2000)
    window = source[window_start:shifted_start]
    comment_start = window.rfind("/**")
    if comment_start == -1:
        return source, False
    comment_end = window.find("*/", comment_start)
    if comment_end == -1:
        return source, False

    absolute_start = window_start + comment_start
    absolute_end = window_start + comment_end + 2
    comment = source[absolute_start:absolute_end]
    updated = re.sub(rf"(@param\s+){re.escape(old_name)}\b", rf"\1{new_name}", comment)
    if updated == comment:
        return source, False
    return source[:absolute_start] + updated + source[absolute_end:], True
