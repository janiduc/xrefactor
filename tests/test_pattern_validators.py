"""
Unit tests for src.transformer.pattern_validators - one clearly-passing and
one clearly-failing case per pattern.
"""

import pytest

from src.transformer.pattern_validators import validate


class TestExtractMethod:
    def test_shorter_distinct_method_passes(self):
        before = "void checkInventoryLevels() { if (a) { doA(); } if (b) { doB(); } if (c) { doC(); } }"
        after = "void restock() { doA(); }"
        assert validate("extract_method", before, after)["passed"]

    def test_unparseable_after_fails(self):
        before = "void checkInventoryLevels() { if (a) { doA(); } }"
        after = "this is not java at all {{{"
        assert not validate("extract_method", before, after)["passed"]

    def test_identical_text_fails(self):
        code = "void foo() { doA(); }"
        assert not validate("extract_method", code, code)["passed"]


class TestRenameVariable:
    def test_single_swapped_identifier_passes(self):
        before = "int itemsToReorder = compute(); return itemsToReorder;"
        after = "int reorderCount = compute(); return reorderCount;"
        assert validate("rename_variable", before, after)["passed"]

    def test_wholesale_rewrite_fails(self):
        before = "int x = compute(); return x;"
        after = "String totallyDifferentApproach = doSomethingElseEntirely(); logResult(totallyDifferentApproach);"
        assert not validate("rename_variable", before, after)["passed"]


class TestRemoveDeadCode:
    def test_meaningful_shrink_passes(self):
        before = "void unused() { doA(); doB(); doC(); doD(); doE(); }"
        after = ""
        assert validate("remove_dead_code", before, after)["passed"]

    def test_no_shrink_fails(self):
        before = "void unused() { doA(); }"
        after = "void unused() { doA(); doB(); }"
        assert not validate("remove_dead_code", before, after)["passed"]


class TestConsolidateDuplicateCode:
    def test_valid_distinct_method_passes(self):
        before = "void a() { x = 1; }"
        after = "void shared() { x = 1; }"
        assert validate("consolidate_duplicate_code", before, after)["passed"]

    def test_unparseable_fails(self):
        before = "void a() { x = 1; }"
        after = "@#$%^ not code"
        assert not validate("consolidate_duplicate_code", before, after)["passed"]


class TestSimplifyCondition:
    def test_fewer_boolean_ops_passes(self):
        before = "if (a && b && c && d) { doIt(); }"
        after = "if (isValid(a, b, c, d)) { doIt(); }"
        assert validate("simplify_condition", before, after)["passed"]

    def test_more_boolean_ops_fails(self):
        before = "if (a) { doIt(); }"
        after = "if (a && b && c) { doIt(); }"
        assert not validate("simplify_condition", before, after)["passed"]


class TestSplitClass:
    def test_new_class_passes(self):
        before = "class Big { void a() {} }"
        after = "class Extracted { void b() {} }"
        assert validate("split_class", before, after)["passed"]

    def test_not_a_class_fails(self):
        before = "class Big { void a() {} }"
        after = "void notAClass() {}"
        assert not validate("split_class", before, after)["passed"]


class TestExtractInterface:
    def test_interface_passes(self):
        before = "class Impl { void a() {} }"
        after = "interface Contract { void a(); }"
        assert validate("extract_interface", before, after)["passed"]

    def test_class_not_interface_fails(self):
        before = "class Impl { void a() {} }"
        after = "class StillAClass { void a() {} }"
        assert not validate("extract_interface", before, after)["passed"]


class TestReduceCoupling:
    def test_fewer_qualified_calls_passes(self):
        before = "void a() { serviceA.doX(); serviceB.doY(); serviceC.doZ(); }"
        after = "void a() { facade.doAll(); }"
        assert validate("reduce_coupling", before, after)["passed"]

    def test_more_qualified_calls_fails(self):
        before = "void a() { facade.doAll(); }"
        after = "void a() { serviceA.doX(); serviceB.doY(); serviceC.doZ(); }"
        assert not validate("reduce_coupling", before, after)["passed"]


class TestMoveClass:
    def test_valid_distinct_declaration_passes(self):
        before = "class Foo { void a() {} }"
        after = "class Foo { void a() { doExtra(); } }"
        assert validate("move_class", before, after)["passed"]

    def test_identical_fails(self):
        code = "class Foo { void a() {} }"
        assert not validate("move_class", code, code)["passed"]


class TestImproveNaming:
    def test_swapped_name_passes(self):
        before = "void doStuff() { doStuff(); }"
        after = "void validateInput() { validateInput(); }"
        assert validate("improve_naming", before, after)["passed"]


class TestJunkIsRejected:
    """Regression pins for output that previously scored as 'passed'.

    These exact strings came out of the real trained model. The first one passed
    `validate_reduce_coupling` because its only check was a qualified-call count
    of 0 -> 0, which unparseable text satisfies trivially.
    """

    REAL_MODEL_JUNK = [
        "public boolean is  private final Collection<>;",
        "\tAnalysisState<>;",
        "\tAnalysis\n  private final Collection\n  private final Collection\n  private final Collection",
    ]

    @pytest.mark.parametrize("junk", REAL_MODEL_JUNK)
    @pytest.mark.parametrize("pattern", ["reduce_coupling", "remove_dead_code",
                                          "rename_variable", "extract_method"])
    def test_unparseable_output_never_passes(self, pattern, junk):
        before = "void f() { serviceA.doX(); serviceB.doY(); }"
        assert not validate(pattern, before, junk)["passed"]

    def test_empty_output_only_counts_for_deletion(self):
        before = "private void dead() { }"
        assert validate("remove_dead_code", before, "")["passed"]
        assert not validate("rename_variable", before, "")["passed"]
        assert not validate("extract_method", before, "")["passed"]


class TestRenameValidatorIsStructural:
    def test_rewrite_sharing_names_is_rejected(self):
        """Same identifiers, different structure - the old heuristic let this
        through because the identifier sets barely changed."""
        before = "int total = compute(); return total;"
        after = "int total = compute(); if (total > 0) { return total; } return 0;"
        assert not validate("rename_variable", before, after)["passed"]

    def test_two_different_names_changed_is_rejected(self):
        before = "int a = 1; int b = 2; return a + b;"
        after = "int x = 1; int y = 2; return x + y;"
        result = validate("rename_variable", before, after)
        assert not result["passed"]
        assert "exactly one" in result["reason"]

    def test_pure_rename_reports_the_pair(self):
        before = "int tmp = compute(); return tmp;"
        after = "int seed = compute(); return seed;"
        result = validate("rename_variable", before, after)
        assert result["passed"], result
        assert result["renamed_from"] == "tmp"
        assert result["renamed_to"] == "seed"
        assert result["occurrences"] == 2


class TestUnknownPattern:
    def test_unknown_pattern_name_fails_gracefully(self):
        result = validate("not_a_real_pattern", "a", "b")
        assert not result["passed"]
        assert "unknown pattern" in result["reason"]
