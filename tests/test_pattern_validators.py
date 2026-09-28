"""
Unit tests for src.transformer.pattern_validators - one clearly-passing and
one clearly-failing case per pattern.
"""

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


class TestUnknownPattern:
    def test_unknown_pattern_name_fails_gracefully(self):
        result = validate("not_a_real_pattern", "a", "b")
        assert not result["passed"]
        assert "unknown pattern" in result["reason"]
