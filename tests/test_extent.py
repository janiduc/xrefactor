"""
Tests for src/refactor/offsets.py and src/refactor/extent.py.

The adversarial fixtures here are the whole reason extent finding is done over
javalang's TOKEN stream rather than by scanning characters: braces inside
character literals, string literals with escaped quotes, line comments, block
comments and annotation array arguments must never be mistaken for a member's
body braces.
"""

import glob
import os

import javalang
import pytest
from javalang.tokenizer import Separator

from src.refactor.extent import (
    Extent,
    brace_match,
    expand_start_backwards,
    member_extent,
    top_level_statement_extents,
)
from src.refactor.offsets import offset_to_pos, pos_to_offset, tokenize_cached

SQ = chr(39)
DQ = chr(34)
BS = chr(92)


def _brace_tokens_balanced(text: str) -> bool:
    """True when `{`/`}` SEPARATOR tokens balance. Deliberately token-based:
    corpus code contains commented-out and quoted braces that a character
    count would miscount. Wraps the text so a bare member still lexes."""
    tokens = tokenize_cached("class __W__ { " + text + " }")
    if tokens is None:
        return False
    depth = 0
    for tok in tokens:
        if isinstance(tok, Separator):
            if tok.value == "{":
                depth += 1
            elif tok.value == "}":
                depth -= 1
                if depth < 0:
                    return False
    return depth == 0


def _method(source, name):
    tree = javalang.parse.parse(source)
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        if node.name == name:
            return node
    raise AssertionError(f"method {name} not found")


class TestOffsets:
    def test_pos_to_offset_lands_on_token_text(self):
        source = "class A {\n  int x = 1;\n}"
        tokens = tokenize_cached(source)
        for tok in tokens:
            offset = pos_to_offset(source, tok.position[0], tok.position[1])
            assert source[offset:offset + len(tok.value)] == tok.value

    def test_offset_to_pos_round_trips(self):
        source = "class A {\n  int x = 1;\n  int y = 2;\n}"
        for offset in range(len(source)):
            line, col = offset_to_pos(source, offset)
            assert pos_to_offset(source, line, col) == offset

    def test_unlexable_source_returns_none(self):
        # Unterminated string literal: the lexer must fail, not raise through.
        assert tokenize_cached('class A { String s = "oops; }') is None


class TestBraceMatchingIgnoresLiteralsAndComments:
    def test_braces_in_char_string_and_comments_are_not_separators(self):
        source = (
            "class A { void f() { char c = " + SQ + "{" + SQ + "; "
            "String s = " + DQ + "a{b" + BS + DQ + "c" + DQ + "; "
            "/* } { */ // }\n int x; } }"
        )
        tokens = tokenize_cached(source)
        braces = [t.value for t in tokens if t.value in ("{", "}")]
        # Only the class body and the method body braces are real.
        assert braces == ["{", "{", "}", "}"]

    def test_method_extent_survives_braces_in_literals(self):
        source = (
            "class A {\n"
            "  void f() {\n"
            "    char c = " + SQ + "}" + SQ + ";\n"
            "    String s = " + DQ + "}}}" + DQ + ";\n"
            "    // }\n"
            "    /* } */\n"
            "  }\n"
            "  void after() {}\n"
            "}\n"
        )
        extent = member_extent(source, _method(source, "f"))
        text = extent.text(source)
        assert text.startswith("void f()")
        assert text.rstrip().endswith("}")
        assert "void after" not in text

    def test_brace_match_returns_none_when_unbalanced(self):
        source = "class A { void f() { "
        tokens = tokenize_cached(source)
        open_idx = next(i for i, t in enumerate(tokens) if t.value == "{")
        # The class brace never closes in this truncated source.
        assert brace_match(tokens, open_idx) is None


class TestMemberExtent:
    def test_includes_modifiers_and_annotations(self):
        source = (
            "class A {\n"
            "  @Override\n"
            "  @SuppressWarnings({" + DQ + "a" + DQ + "," + DQ + "b" + DQ + "})\n"
            "  public static final int f() {\n"
            "    return 1;\n"
            "  }\n"
            "}\n"
        )
        text = member_extent(source, _method(source, "f")).text(source)
        assert text.startswith("@Override")
        assert "@SuppressWarnings" in text
        assert "public static final int f()" in text
        assert text.rstrip().endswith("}")

    def test_annotation_array_braces_are_not_the_body(self):
        """The `{` of `@SuppressWarnings({...})` sits inside `(...)`, so a
        paren-depth-0 rule must skip it and find the real body brace."""
        source = (
            "class A {\n"
            "  @SuppressWarnings({" + DQ + "x" + DQ + "})\n"
            "  void f() { int a = 1; }\n"
            "  void g() {}\n"
            "}\n"
        )
        text = member_extent(source, _method(source, "f")).text(source)
        assert "int a = 1;" in text
        assert "void g" not in text

    def test_excluding_modifiers_starts_at_declaration(self):
        source = "class A {\n  public static void f() { }\n}\n"
        node = _method(source, "f")
        with_mods = member_extent(source, node, include_modifiers=True).text(source)
        without = member_extent(source, node, include_modifiers=False).text(source)
        assert with_mods.startswith("public static")
        assert without.startswith("void f()")

    def test_abstract_method_without_body_ends_at_semicolon(self):
        source = "abstract class A {\n  protected abstract int f(int x);\n  void g() {}\n}\n"
        text = member_extent(source, _method(source, "f")).text(source)
        assert text == "protected abstract int f(int x);"

    def test_generic_return_type_and_throws(self):
        source = (
            "import java.util.*;\n"
            "class A {\n"
            "  public <T> Map<String, List<T>> f(int x) throws java.io.IOException {\n"
            "    return null;\n"
            "  }\n"
            "}\n"
        )
        text = member_extent(source, _method(source, "f")).text(source)
        assert text.startswith("public <T> Map<String, List<T>> f(")
        assert "throws java.io.IOException" in text
        assert text.rstrip().endswith("}")

    def test_nested_and_anonymous_classes_and_lambdas(self):
        source = (
            "class A {\n"
            "  void f() {\n"
            "    Runnable r = new Runnable() { public void run() { int q = 1; } };\n"
            "    Runnable s = () -> { int w = 2; };\n"
            "    class Local { void m() { } }\n"
            "  }\n"
            "  void sentinel() {}\n"
            "}\n"
        )
        text = member_extent(source, _method(source, "f")).text(source)
        assert "class Local" in text
        assert "sentinel" not in text
        assert _brace_tokens_balanced(text)

    def test_field_extent_ends_at_semicolon(self):
        source = "class A {\n  private int field = 3;\n  void f() {}\n}\n"
        tree = javalang.parse.parse(source)
        field = next(n for _, n in tree.filter(javalang.tree.FieldDeclaration))
        assert member_extent(source, field).text(source) == "private int field = 3;"

    def test_expand_start_backwards_stops_at_previous_member(self):
        source = "class A {\n  void a() {}\n  void b() {}\n}\n"
        tokens = tokenize_cached(source)
        node = _method(source, "b")
        decl_idx = next(
            i for i, t in enumerate(tokens)
            if (t.position[0], t.position[1]) == (node.position[0], node.position[1])
        )
        assert expand_start_backwards(tokens, decl_idx) == decl_idx


class TestTopLevelStatementExtents:
    def test_aligns_with_body_and_slices_each_statement(self):
        source = (
            "class A {\n"
            "  void f() {\n"
            "    int a = 1;\n"
            "    if (a > 0) {\n"
            "      a++;\n"
            "    }\n"
            "    foo(a);\n"
            "  }\n"
            "}\n"
        )
        node = _method(source, "f")
        extents = top_level_statement_extents(source, node)
        assert len(extents) == len(node.body) == 3
        texts = [e.text(source) for e in extents]
        assert texts[0] == "int a = 1;"
        assert texts[1].startswith("if (a > 0) {") and texts[1].endswith("}")
        assert texts[2] == "foo(a);"

    def test_extents_are_disjoint_and_ordered(self):
        source = "class A {\n  void f() {\n    int a = 1;\n    int b = 2;\n    int c = 3;\n  }\n}\n"
        extents = top_level_statement_extents(source, _method(source, "f"))
        for earlier, later in zip(extents, extents[1:]):
            assert earlier.end_offset <= later.start_offset

    def test_empty_body_returns_none(self):
        source = "class A {\n  void f() {}\n}\n"
        assert top_level_statement_extents(source, _method(source, "f")) is None


@pytest.mark.slow
class TestCorpusProperties:
    """Every method extent in the real corpus must be well-formed. This is the
    guard that makes the engine's later edits trustworthy: if an extent is
    wrong, every splice built on it is wrong."""

    @staticmethod
    def _corpus_files(limit=400):
        pattern = os.path.join("..", "Data", "**", "*.java")
        files = sorted(glob.glob(pattern, recursive=True))
        return files[:limit]

    def test_method_extents_are_well_formed(self):
        files = self._corpus_files()
        assert files, "no corpus files found under ../Data"

        checked = 0
        skipped_unparseable = 0
        for path in files:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                source = handle.read()
            try:
                tree = javalang.parse.parse(source)
            except Exception:
                skipped_unparseable += 1
                continue

            per_file = []
            for _, node in tree.filter(javalang.tree.MethodDeclaration):
                extent = member_extent(source, node)
                if extent is None:
                    continue
                text = extent.text(source)
                # Count brace SEPARATOR TOKENS, not raw characters: real corpus
                # methods contain commented-out braces (`//        }`) and braces
                # inside string/char literals, which a character count would
                # wrongly treat as unbalanced. This is the same distinction the
                # implementation relies on.
                assert _brace_tokens_balanced(text), f"unbalanced brace tokens in {path}:{node.name}"
                assert text.rstrip()[-1] in "};", f"extent ends at neither brace nor semicolon in {path}:{node.name}"
                assert 0 <= extent.start_offset < extent.end_offset <= len(source)
                per_file.append((extent, node.name))
                checked += 1

            # Sibling extents must be disjoint or properly nested, never partially overlapping.
            for i in range(len(per_file)):
                for j in range(i + 1, len(per_file)):
                    a, a_name = per_file[i]
                    b, b_name = per_file[j]
                    disjoint = a.end_offset <= b.start_offset or b.end_offset <= a.start_offset
                    nested = (a.start_offset <= b.start_offset and b.end_offset <= a.end_offset) or (
                        b.start_offset <= a.start_offset and a.end_offset <= b.end_offset
                    )
                    assert disjoint or nested, f"partial overlap {a_name}/{b_name} in {path}"

        assert checked > 500, f"expected to check many methods, only checked {checked}"
        print(f"\nchecked {checked} method extents across {len(files)} files "
              f"({skipped_unparseable} unparseable, skipped)")
