"""
Unit tests for src.cpg.smell_detector - one positive and one negative case
per detector, built on small synthetic CPGs so each test is independent of
any specific mined dataset.
"""

import pytest

from src.cpg.cpg_builder import CodeEdge, CodePropertyGraph
from src.cpg.smell_detector import SmellDetector


class _TestConfig:
    """Minimal config stand-in with tight thresholds so small synthetic
    examples can trip the detectors without needing hundreds of nodes."""
    def __init__(self, overrides=None):
        self.overrides = overrides or {}

    def get(self, key, default=None):
        return self.overrides.get(key, default)


def _add_statement_chain(cpg, method_id, file_path, method_name, names_and_snippets, first_kind="entry"):
    """Wires a linear control_flow chain of statement nodes hanging off method_id."""
    prev_id = method_id
    edge_kind = first_kind
    for i, (stmt_name, snippet) in enumerate(names_and_snippets):
        stmt_id = f"stmt_{file_path}_{method_name}_{i}"
        cpg._add_node(node_id=stmt_id, node_type="statement", name=stmt_name,
                       file_path=file_path, line_number=10 + i, code_snippet=snippet)
        cpg.edges.append(CodeEdge(source_id=prev_id, target_id=stmt_id, edge_type="control_flow",
                                   metadata={"kind": edge_kind}))
        edge_kind = "sequence"
        prev_id = stmt_id


class TestLongMethod:
    def test_flags_method_over_threshold(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="m1", node_type="method", name="doWork", file_path="A.java",
                       line_number=1, code_snippet="void doWork() {")
        _add_statement_chain(cpg, "m1", "A.java", "doWork",
                              [("StatementExpression", f"s{i};") for i in range(20)])

        smells = SmellDetector(_TestConfig({"smells.long_method_statement_threshold": 15})).detect(cpg)
        long_method_smells = [s for s in smells if s["smell_type"] == "long_method"]
        assert len(long_method_smells) == 1
        assert long_method_smells[0]["node_id"] == "m1"
        assert long_method_smells[0]["refactoring_pattern"] == "extract_method"
        assert long_method_smells[0]["metric_evidence"]["statement_count"] == 20

    def test_short_method_not_flagged(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="m1", node_type="method", name="getX", file_path="A.java",
                       line_number=1, code_snippet="int getX() {")
        _add_statement_chain(cpg, "m1", "A.java", "getX", [("ReturnStatement", "return x;")])

        smells = SmellDetector(_TestConfig({"smells.long_method_statement_threshold": 15})).detect(cpg)
        assert not [s for s in smells if s["smell_type"] == "long_method"]


class TestDuplicateCode:
    def test_flags_near_identical_methods(self):
        cpg = CodePropertyGraph(language="java")
        shared_lines = [("StatementExpression", f"x = {i};") for i in range(5)]
        cpg._add_node(node_id="m1", node_type="method", name="a", file_path="A.java", line_number=1, code_snippet="")
        cpg._add_node(node_id="m2", node_type="method", name="b", file_path="A.java", line_number=20, code_snippet="")
        _add_statement_chain(cpg, "m1", "A.java", "a", shared_lines)
        _add_statement_chain(cpg, "m2", "A.java", "b", shared_lines)

        smells = SmellDetector(_TestConfig({
            "smells.duplicate_code_min_statements": 3,
            "smells.duplicate_code_similarity_threshold": 0.7,
        })).detect(cpg)
        dup_smells = [s for s in smells if s["smell_type"] == "duplicate_code"]
        assert len(dup_smells) == 1
        assert dup_smells[0]["refactoring_pattern"] == "consolidate_duplicate_code"

    def test_dissimilar_methods_not_flagged(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="m1", node_type="method", name="a", file_path="A.java", line_number=1, code_snippet="")
        cpg._add_node(node_id="m2", node_type="method", name="b", file_path="A.java", line_number=20, code_snippet="")
        _add_statement_chain(cpg, "m1", "A.java", "a", [("StatementExpression", f"x = {i};") for i in range(5)])
        _add_statement_chain(cpg, "m2", "A.java", "b", [("StatementExpression", f"y = {i}*2;") for i in range(5)])

        smells = SmellDetector(_TestConfig({
            "smells.duplicate_code_min_statements": 3,
            "smells.duplicate_code_similarity_threshold": 0.7,
        })).detect(cpg)
        assert not [s for s in smells if s["smell_type"] == "duplicate_code"]


class TestDeadCode:
    def test_flags_never_called_method(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="method_A.java_unused", node_type="method", name="unused",
                       file_path="A.java", line_number=1, code_snippet="")
        smells = SmellDetector().detect(cpg)
        dead = [s for s in smells if s["smell_type"] == "dead_code"]
        assert len(dead) == 1
        assert dead[0]["refactoring_pattern"] == "remove_dead_code"

    def test_called_method_not_flagged(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="method_A.java_caller", node_type="method", name="caller",
                       file_path="A.java", line_number=1, code_snippet="")
        cpg._add_node(node_id="method_A.java_used", node_type="method", name="used",
                       file_path="A.java", line_number=5, code_snippet="")
        cpg.edges.append(CodeEdge(source_id="method_A.java_caller", target_id="method_A.java_used",
                                   edge_type="calls"))
        smells = SmellDetector().detect(cpg)
        dead_ids = {s["node_id"] for s in smells if s["smell_type"] == "dead_code"}
        assert "method_A.java_used" not in dead_ids

    def test_main_not_flagged_despite_no_callers(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="method_A.java_main", node_type="method", name="main",
                       file_path="A.java", line_number=1, code_snippet="")
        smells = SmellDetector().detect(cpg)
        assert not [s for s in smells if s["smell_type"] == "dead_code"]


class TestComplexCondition:
    def test_flags_many_branches(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="m1", node_type="method", name="branchy", file_path="A.java",
                       line_number=1, code_snippet="")
        _add_statement_chain(cpg, "m1", "A.java", "branchy",
                              [("IfStatement", "if (a) {") for _ in range(6)])

        smells = SmellDetector(_TestConfig({"smells.complex_condition_branch_threshold": 5})).detect(cpg)
        complex_smells = [s for s in smells if s["smell_type"] == "complex_condition"]
        assert len(complex_smells) == 1
        assert complex_smells[0]["refactoring_pattern"] == "simplify_condition"

    def test_flags_many_boolean_operators(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="m1", node_type="method", name="cond", file_path="A.java",
                       line_number=1, code_snippet="")
        _add_statement_chain(cpg, "m1", "A.java", "cond",
                              [("IfStatement", "if (a && b && c && d && e) {")])

        smells = SmellDetector(_TestConfig({"smells.complex_condition_boolean_op_threshold": 3})).detect(cpg)
        assert [s for s in smells if s["smell_type"] == "complex_condition"]

    def test_simple_method_not_flagged(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="m1", node_type="method", name="simple", file_path="A.java",
                       line_number=1, code_snippet="")
        _add_statement_chain(cpg, "m1", "A.java", "simple", [("IfStatement", "if (a) {")])

        smells = SmellDetector().detect(cpg)
        assert not [s for s in smells if s["smell_type"] == "complex_condition"]


class TestGodClassAndCoupling:
    def test_flags_large_class(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="class_A.java_Big", node_type="class", name="Big",
                       file_path="A.java", line_number=1, code_snippet="")
        for i in range(25):
            cpg._add_node(node_id=f"method_A.java_m{i}", node_type="method", name=f"m{i}",
                           file_path="A.java", line_number=10 + i, code_snippet="")

        smells = SmellDetector(_TestConfig({"smells.god_class_member_threshold": 20})).detect(cpg)
        god_smells = [s for s in smells if s["smell_type"] == "god_class"]
        assert len(god_smells) == 1
        assert god_smells[0]["refactoring_pattern"] == "split_class"

    def test_small_class_not_flagged(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="class_A.java_Small", node_type="class", name="Small",
                       file_path="A.java", line_number=1, code_snippet="")
        cpg._add_node(node_id="method_A.java_m0", node_type="method", name="m0",
                      file_path="A.java", line_number=10, code_snippet="")
        smells = SmellDetector(_TestConfig({"smells.god_class_member_threshold": 20})).detect(cpg)
        assert not [s for s in smells if s["smell_type"] == "god_class"]

    def test_flags_high_coupling(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="class_A.java_Hub", node_type="class", name="Hub",
                       file_path="A.java", line_number=1, code_snippet="")
        cpg._add_node(node_id="method_A.java_hub", node_type="method", name="hub",
                       file_path="A.java", line_number=5, code_snippet="")
        for i in range(12):
            other_file = f"Other{i}.java"
            cpg._add_node(node_id=f"method_{other_file}_x", node_type="method", name="x",
                           file_path=other_file, line_number=1, code_snippet="")
            cpg.edges.append(CodeEdge(source_id="method_A.java_hub", target_id=f"method_{other_file}_x",
                                       edge_type="calls"))

        smells = SmellDetector(_TestConfig({"smells.high_coupling_file_threshold": 10})).detect(cpg)
        coupling_smells = [s for s in smells if s["smell_type"] == "high_coupling"]
        assert len(coupling_smells) == 1
        assert coupling_smells[0]["refactoring_pattern"] == "reduce_coupling"


class TestPoorNaming:
    def test_flags_generic_variable_name(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="v1", node_type="variable", name="temp", file_path="A.java",
                       line_number=1, code_snippet="")
        smells = SmellDetector().detect(cpg)
        naming = [s for s in smells if s["smell_type"] == "poor_naming" and s["refactoring_pattern"] == "rename_variable"]
        assert len(naming) == 1

    def test_conventional_loop_counter_not_flagged(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="v1", node_type="variable", name="i", file_path="A.java",
                       line_number=1, code_snippet="")
        smells = SmellDetector().detect(cpg)
        assert not [s for s in smells if s["smell_type"] == "poor_naming"]

    def test_descriptive_variable_name_not_flagged(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="v1", node_type="variable", name="userAccountBalance", file_path="A.java",
                       line_number=1, code_snippet="")
        smells = SmellDetector().detect(cpg)
        assert not [s for s in smells if s["smell_type"] == "poor_naming"]

    def test_flags_generic_method_name(self):
        cpg = CodePropertyGraph(language="java")
        cpg._add_node(node_id="method_A.java_do", node_type="method", name="do", file_path="A.java",
                       line_number=1, code_snippet="")
        smells = SmellDetector().detect(cpg)
        naming = [s for s in smells if s["smell_type"] == "poor_naming" and s["refactoring_pattern"] == "improve_naming"]
        assert len(naming) == 1
