"""
Position <-> offset conversion and cached tokenization.

javalang reports positions as 1-based (line, column) pairs; every edit in this
package is applied as a character-offset splice, so one reliable conversion
lives here rather than being re-derived at each call site.
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import javalang
from javalang.tokenizer import LexerError


@lru_cache(maxsize=32)
def line_starts(source: str) -> Tuple[int, ...]:
    """Character offset at which each line begins (index 0 == line 1)."""
    starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            starts.append(i + 1)
    return tuple(starts)


def pos_to_offset(source: str, line: int, column: int) -> int:
    """1-based (line, column) -> 0-based character offset.

    javalang's columns are 1-based: `source[offset:offset+len(tok.value)]`
    equals the token's text.
    """
    starts = line_starts(source)
    if line < 1 or line > len(starts):
        raise ValueError(f"line {line} out of range (source has {len(starts)} lines)")
    return starts[line - 1] + (column - 1)


def offset_to_pos(source: str, offset: int) -> Tuple[int, int]:
    """0-based character offset -> 1-based (line, column)."""
    starts = line_starts(source)
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1, offset - starts[lo] + 1


@lru_cache(maxsize=32)
def tokenize_cached(source: str) -> Optional[Tuple]:
    """Tokenize once per source string. Returns None if the source does not lex.

    Tokenization is the backbone of extent finding: javalang's lexer already
    resolves string literals, character literals (including `'{'` and `'\\''`)
    and both comment forms, so braces appearing inside them never surface as
    `Separator` tokens and cannot confuse brace matching.
    """
    try:
        return tuple(javalang.tokenizer.tokenize(source))
    except (LexerError, Exception):  # javalang also raises bare ValueError/IndexError
        return None


def token_offsets(source: str, tokens) -> List[int]:
    """Character offset of each token, parallel to `tokens`."""
    return [pos_to_offset(source, t.position[0], t.position[1]) for t in tokens]


def token_index_at(tokens, line: int, column: int) -> Optional[int]:
    """Index of the token starting exactly at (line, column), else None."""
    for i, t in enumerate(tokens):
        if t.position[0] == line and t.position[1] == column:
            return i
    return None


def first_token_index_at_or_after(tokens, line: int, column: int) -> Optional[int]:
    """Index of the first token starting at or after (line, column)."""
    for i, t in enumerate(tokens):
        if (t.position[0], t.position[1]) >= (line, column):
            return i
    return None
