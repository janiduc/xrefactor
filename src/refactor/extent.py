"""
Source extents for Java members and statements.

javalang gives declarations a START position only - there is no `end_position` -
so the end of a member is found by matching its body braces over the TOKEN
stream. That is sound because javalang's lexer never emits a `Separator` token
for a brace inside a string literal, a character literal (`'{'`), or either
comment form; verified against adversarial fixtures in tests/test_extent.py.

Two details the token-level approach gets right and a naive text scan does not:
  * Annotation array arguments (`@SuppressWarnings({"a","b"})`) DO produce real
    `{`/`}` Separator tokens - but always inside `(...)`, so a paren-depth-0
    rule skips them.
  * Generics (`Map<String, List<T>>`) are `Operator` tokens, not brackets, so
    they never perturb depth tracking.

A declaration's `position` sits AFTER its modifiers and annotations (verified:
`public static final <T> ... f()` reports the `<` of `<T>`), so callers wanting
the full member source expand the start backwards over `Modifier` tokens and
annotations.
"""

from dataclasses import dataclass
from typing import List, Optional

from javalang.tokenizer import Annotation, Identifier, Modifier, Separator

from src.refactor.offsets import (
    first_token_index_at_or_after,
    offset_to_pos,
    pos_to_offset,
    token_index_at,
    tokenize_cached,
)


@dataclass(frozen=True)
class Extent:
    """A half-open character range `[start_offset, end_offset)` over the source."""

    start_offset: int
    end_offset: int
    start_line: int
    end_line: int
    body_open_offset: Optional[int] = None  # offset of the member's '{', if it has a body

    def text(self, source: str) -> str:
        return source[self.start_offset:self.end_offset]

    def to_dict(self) -> dict:
        return {
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "body_open_offset": self.body_open_offset,
        }


def _tok_offset(source: str, token) -> int:
    return pos_to_offset(source, token.position[0], token.position[1])


def brace_match(tokens, open_idx: int) -> Optional[int]:
    """Index of the `}` matching the `{` at `open_idx`, or None if unbalanced."""
    if tokens[open_idx].value != "{":
        return None
    depth = 0
    for i in range(open_idx, len(tokens)):
        tok = tokens[i]
        if not isinstance(tok, Separator):
            continue
        if tok.value == "{":
            depth += 1
        elif tok.value == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def paren_match_backwards(tokens, close_idx: int) -> Optional[int]:
    """Index of the `(` matching the `)` at `close_idx`, scanning backwards."""
    if tokens[close_idx].value != ")":
        return None
    depth = 0
    for i in range(close_idx, -1, -1):
        tok = tokens[i]
        if not isinstance(tok, Separator):
            continue
        if tok.value == ")":
            depth += 1
        elif tok.value == "(":
            depth -= 1
            if depth == 0:
                return i
    return None


def _skip_qualified_name_backwards(tokens, idx: int) -> int:
    """Given `idx` at the last Identifier of a possibly-qualified name (`a.b.C`),
    return the index immediately before the name's first Identifier."""
    if idx < 0 or not isinstance(tokens[idx], Identifier):
        return idx
    i = idx
    while i - 2 >= 0 and isinstance(tokens[i - 1], Separator) and tokens[i - 1].value == "." \
            and isinstance(tokens[i - 2], Identifier):
        i -= 2
    return i - 1


def expand_start_backwards(tokens, decl_idx: int) -> int:
    """Walk backwards from a declaration's first token over its modifiers and
    annotations, returning the token index the full declaration starts at.

    Handles `public static final`, `@Override`, `@a.b.Anno`, and
    `@SuppressWarnings({"a","b"})` (whose parenthesised argument list is matched
    back as a unit, so its array braces cannot be mistaken for a body)."""
    idx = decl_idx
    while idx > 0:
        prev = idx - 1
        tok = tokens[prev]

        if isinstance(tok, Modifier):
            idx = prev
            continue

        # Annotation with arguments: ... @Name( ... ) <decl>
        if isinstance(tok, Separator) and tok.value == ")":
            open_idx = paren_match_backwards(tokens, prev)
            if open_idx is None or open_idx == 0:
                break
            before_name = _skip_qualified_name_backwards(tokens, open_idx - 1)
            if before_name >= 0 and isinstance(tokens[before_name], Annotation):
                idx = before_name
                continue
            break

        # Marker annotation: ... @Name <decl>
        if isinstance(tok, Identifier):
            before_name = _skip_qualified_name_backwards(tokens, prev)
            if before_name >= 0 and isinstance(tokens[before_name], Annotation):
                idx = before_name
                continue
            break

        break
    return idx


def _find_body_open(tokens, start_idx: int) -> Optional[int]:
    """First `{` at paren/bracket depth 0 at or after `start_idx`, or None if a
    depth-0 `;` is reached first (an abstract/native method or a field)."""
    paren = 0
    bracket = 0
    for i in range(start_idx, len(tokens)):
        tok = tokens[i]
        if not isinstance(tok, Separator):
            continue
        v = tok.value
        if v == "(":
            paren += 1
        elif v == ")":
            paren -= 1
        elif v == "[":
            bracket += 1
        elif v == "]":
            bracket -= 1
        elif v == ";" and paren == 0 and bracket == 0:
            return None
        elif v == "{" and paren == 0 and bracket == 0:
            return i
    return None


def _find_semicolon(tokens, start_idx: int) -> Optional[int]:
    """First `;` at paren/bracket/brace depth 0 at or after `start_idx`."""
    paren = bracket = brace = 0
    for i in range(start_idx, len(tokens)):
        tok = tokens[i]
        if not isinstance(tok, Separator):
            continue
        v = tok.value
        if v == "(":
            paren += 1
        elif v == ")":
            paren -= 1
        elif v == "[":
            bracket += 1
        elif v == "]":
            bracket -= 1
        elif v == "{":
            brace += 1
        elif v == "}":
            brace -= 1
        elif v == ";" and paren == 0 and bracket == 0 and brace == 0:
            return i
    return None


def member_extent(source: str, node, include_modifiers: bool = True) -> Optional[Extent]:
    """Full source extent of a declaration (method, constructor, class, field).

    Returns None when the source does not lex, the declaration's position cannot
    be located in the token stream, or its braces are unbalanced - callers treat
    None as "refuse to touch this file" rather than guessing.
    """
    if node.position is None:
        return None
    tokens = tokenize_cached(source)
    if tokens is None:
        return None

    decl_idx = token_index_at(tokens, node.position[0], node.position[1])
    if decl_idx is None:
        decl_idx = first_token_index_at_or_after(tokens, node.position[0], node.position[1])
    if decl_idx is None:
        return None

    # Search for the body from the declaration proper, never from the expanded
    # start: an annotation's `{...}` sits before `decl_idx` and must stay excluded.
    body_open_idx = _find_body_open(tokens, decl_idx)
    if body_open_idx is not None:
        close_idx = brace_match(tokens, body_open_idx)
        if close_idx is None:
            return None
        end_token = tokens[close_idx]
        body_open_offset = _tok_offset(source, tokens[body_open_idx])
    else:
        semi_idx = _find_semicolon(tokens, decl_idx)
        if semi_idx is None:
            return None
        end_token = tokens[semi_idx]
        body_open_offset = None

    start_idx = expand_start_backwards(tokens, decl_idx) if include_modifiers else decl_idx
    start_offset = _tok_offset(source, tokens[start_idx])
    end_offset = _tok_offset(source, end_token) + len(end_token.value)

    return Extent(
        start_offset=start_offset,
        end_offset=end_offset,
        start_line=offset_to_pos(source, start_offset)[0],
        end_line=offset_to_pos(source, end_offset - 1)[0],
        body_open_offset=body_open_offset,
    )


def top_level_statement_extents(source: str, method_node) -> Optional[List[Extent]]:
    """Extents of the statements directly in a method's body, aligned 1:1 with
    `method_node.body`.

    Each statement runs from its own first token up to (not including) the next
    statement's first token, with trailing whitespace trimmed; the last runs to
    just before the body's closing brace. That is exact for the contiguous
    top-level ranges `extract_method` operates on, and avoids needing a general
    statement-end parser. A trailing `// comment` after a statement's `;` is
    attributed to that statement, which is where a reader would expect it.
    """
    if not method_node.body:
        return None
    tokens = tokenize_cached(source)
    if tokens is None:
        return None

    outer = member_extent(source, method_node, include_modifiers=False)
    if outer is None or outer.body_open_offset is None:
        return None

    starts: List[int] = []
    for stmt in method_node.body:
        if getattr(stmt, "position", None) is None:
            return None  # cannot place this statement; refuse rather than guess
        starts.append(pos_to_offset(source, stmt.position[0], stmt.position[1]))

    body_close_offset = outer.end_offset - 1  # offset of the body's '}'
    extents: List[Extent] = []
    for i, start in enumerate(starts):
        raw_end = starts[i + 1] if i + 1 < len(starts) else body_close_offset
        end = raw_end
        while end > start and source[end - 1] in " \t\r\n":
            end -= 1
        extents.append(
            Extent(
                start_offset=start,
                end_offset=end,
                start_line=offset_to_pos(source, start)[0],
                end_line=offset_to_pos(source, max(start, end - 1))[0],
            )
        )
    return extents
