"""
Tests for CodeNode.source_text (the full declaration source fed to Stage 3).

The important property here is a NEGATIVE one: adding source_text must not
change `code_snippet`, because smell_detector's duplicate-code Jaccard
(threshold 0.7) and boolean-operator counts (threshold 3) are calibrated on
single-line statement snippets. Widening code_snippet would silently move
those thresholds.
"""

import os
import tempfile
from collections import Counter

import pytest

from src.cpg.cpg_builder import CodePropertyGraph
from src.cpg.smell_detector import SmellDetector

JAVA = """package demo;

public class Calculator {

    private int total = 0;

    /**
     * Adds two numbers.
     * @param a first
     */
    @Deprecated
    public int add(int a, int b) {
        int sum = a + b;
        if (sum > 100 && a > 0 || b > 0) {
            sum = 100;
        }
        return sum;
    }

    public void noop() {
    }
}
"""


@pytest.fixture
def built_cpg():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "Calculator.java")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(JAVA)
        cpg = CodePropertyGraph(language="java")
        cpg.build_from_directory(tmp)
        yield cpg


def _node(cpg, node_type, name):
    for node in cpg.nodes.values():
        if node.node_type == node_type and node.name == name:
            return node
    raise AssertionError(f"{node_type} {name} not found")


class TestSourceText:
    def test_method_source_text_is_the_whole_method(self, built_cpg):
        node = _node(built_cpg, "method", "add")
        assert node.source_text.count("\n") >= 5, node.source_text
        assert "int sum = a + b;" in node.source_text
        assert node.source_text.rstrip().endswith("}")
        # Modifiers and annotations are included, the javadoc above is not.
        assert "@Deprecated" in node.source_text
        assert "public int add(int a, int b)" in node.source_text

    def test_code_snippet_remains_exactly_one_line(self, built_cpg):
        for node in built_cpg.nodes.values():
            assert "\n" not in node.code_snippet, (
                f"{node.node_type} {node.name} has a multi-line code_snippet; "
                "smell_detector thresholds are calibrated on single lines"
            )

    def test_class_node_has_source_text(self, built_cpg):
        node = _node(built_cpg, "class", "Calculator")
        assert "class Calculator" in node.source_text
        assert "noop" in node.source_text  # whole class body

    def test_extent_metadata_is_recorded(self, built_cpg):
        node = _node(built_cpg, "method", "add")
        extent = node.metadata.get("extent")
        assert extent is not None
        assert extent["start_offset"] < extent["end_offset"]
        assert extent["body_open_offset"] is not None
        assert JAVA[extent["start_offset"]:extent["end_offset"]] == node.source_text

    def test_statement_nodes_have_no_source_text(self, built_cpg):
        statements = [n for n in built_cpg.nodes.values() if n.node_type == "statement"]
        assert statements, "expected statement nodes"
        assert all(n.source_text == "" for n in statements)

    def test_to_dict_exposes_source_text(self, built_cpg):
        as_dict = built_cpg.to_dict()
        method_entries = [v for v in as_dict["nodes"].values() if v["type"] == "method"]
        assert method_entries
        assert any(entry["source_text"] for entry in method_entries)


class TestSmellDetectorUnaffected:
    """The guard on CHANGE A: detector output must be byte-identical to what it
    was before source_text existed, since its thresholds depend on snippets
    staying single-line."""

    def test_smell_counts_depend_only_on_single_line_snippets(self, built_cpg):
        smells = SmellDetector().detect(built_cpg)
        counts = Counter(s["smell_type"] for s in smells)

        # The fixture is written so these are stable and meaningful:
        # `if (sum > 100 && a > 0 || b > 0)` has 2 boolean operators (below the
        # threshold of 3), so complex_condition must NOT fire.
        assert "complex_condition" not in counts

        # Every smell that did fire must reference a node that exists.
        node_ids = set(built_cpg.nodes)
        assert all(s["node_id"] in node_ids for s in smells)

    def test_duplicate_code_still_compares_single_lines(self, built_cpg):
        """duplicate_code builds Jaccard sets from per-statement snippets; if a
        statement snippet ever became multi-line the set would collapse to one
        giant string and the 0.7 threshold would stop meaning anything."""
        method_statements = SmellDetector()._method_statement_map(built_cpg)
        for statements in method_statements.values():
            for statement in statements:
                assert "\n" not in statement.code_snippet
