"""
Tests for cross-file call-graph resolution (src/cpg/call_resolver.py).

The regression these lock down: the builder used to synthesise every call edge
as `method_<CALLER's file>_<invoked name>`, which meant (a) 93% of call edges
pointed at node ids that did not exist and were silently materialised as
phantom NetworkX nodes, and (b) not one single cross-file call edge was ever
produced - in a framework whose premise is cross-file reasoning.

So the two headline assertions here are structural invariants, not thresholds:

    * every edge's endpoints exist in `cpg.nodes` (no phantom nodes), and
    * a call from A.java into B.java produces an edge between those two nodes.

The rest pin each resolution strategy and each deliberate refusal, because a
refusal silently becoming a guess is the failure mode that would quietly
restore the original bug.
"""

import os
import tempfile

import pytest

from src.cpg.call_resolver import (
    CHAINED, PLAIN, QUALIFIED, SUPER, THIS,
    CallResolver, PendingCall, coverage, simple_type_name,
)
from src.cpg.cpg_builder import CodePropertyGraph


def build(files: dict) -> CodePropertyGraph:
    """Build a CPG from {filename: java source}."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, source in files.items():
            path = os.path.join(tmp, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
        cpg = CodePropertyGraph(language="java")
        cpg.build_from_directory(tmp)
        return cpg


def call_edges(cpg):
    return [e for e in cpg.edges if e.edge_type == "calls"]


def find_call(cpg, caller_name: str, target_name: str):
    """The calls edge from the method named `caller_name` to `target_name`."""
    for edge in call_edges(cpg):
        source = cpg.nodes.get(edge.source_id)
        target = cpg.nodes.get(edge.target_id)
        if source and target and source.name == caller_name and target.name == target_name:
            return edge
    return None


def reasons(cpg):
    return {c["reason"] for c in cpg.unresolved_calls}


# --------------------------------------------------------------------------- #
# Structural invariants
# --------------------------------------------------------------------------- #

class TestNoPhantomNodes:
    """NetworkX creates any node it has not seen when you add an edge, so an
    unresolvable target used to inflate the graph behind the statistics."""

    def test_graph_node_count_matches_node_dict(self):
        cpg = build({
            "A.java": "public class A { private B b; void go(){ b.hit(); unknownThing.wat(); } }",
            "B.java": "public class B { void hit(){} }",
        })
        assert cpg.graph.number_of_nodes() == len(cpg.nodes)
        assert cpg.get_statistics()["graph_nodes"] == cpg.get_statistics()["total_nodes"]

    def test_every_edge_endpoint_is_a_real_node(self):
        cpg = build({
            "A.java": """public class A {
                private B b;
                void go(){ b.hit(); System.out.println("x"); helper(); }
                void helper(){}
            }""",
            "B.java": "public class B { void hit(){} }",
        })
        for edge in cpg.edges:
            assert edge.source_id in cpg.nodes, f"dangling source: {edge.source_id}"
            assert edge.target_id in cpg.nodes, f"dangling target: {edge.target_id}"

    def test_unresolved_calls_produce_no_edge_but_are_recorded(self):
        cpg = build({
            "A.java": 'public class A { void go(){ java.util.List<String> xs = null; xs.add("x"); } }',
        })
        # `xs` is a JDK type, so there is nothing in-corpus to point at.
        assert find_call(cpg, "go", "add") is None
        assert any(c["member"] == "add" for c in cpg.unresolved_calls)


# --------------------------------------------------------------------------- #
# Resolution strategies
# --------------------------------------------------------------------------- #

class TestCrossFileResolution:
    def test_field_typed_receiver_resolves_across_files(self):
        cpg = build({
            "Service.java": "public class Service { private Repo repo; void save(){ repo.persist(); } }",
            "Repo.java": "public class Repo { void persist(){} }",
        })
        edge = find_call(cpg, "save", "persist")
        assert edge is not None, "cross-file call via a field's declared type was not resolved"
        assert edge.metadata["resolution"] == "qualifier_type"
        assert edge.metadata["cross_file"] is True
        assert cpg.nodes[edge.target_id].file_path.endswith("Repo.java")

    def test_parameter_typed_receiver_resolves_across_files(self):
        cpg = build({
            "Service.java": "public class Service { void save(Repo repo){ repo.persist(); } }",
            "Repo.java": "public class Repo { void persist(){} }",
        })
        edge = find_call(cpg, "save", "persist")
        assert edge is not None
        assert edge.metadata["cross_file"] is True

    def test_local_variable_receiver_resolves_across_files(self):
        cpg = build({
            "Service.java": "public class Service { void save(){ Repo repo = new Repo(); repo.persist(); } }",
            "Repo.java": "public class Repo { void persist(){} }",
        })
        assert find_call(cpg, "save", "persist") is not None

    def test_enhanced_for_variable_resolves(self):
        """`for (Batch b : batches)` is a VariableDeclaration, not a
        LocalVariableDeclaration - missing it left for-each receivers untyped."""
        cpg = build({
            "Runner.java": """public class Runner {
                void run(java.util.List<Batch> batches){
                    for (Batch b : batches) { b.brew(); }
                }
            }""",
            "Batch.java": "public class Batch { void brew(){} }",
        })
        edge = find_call(cpg, "run", "brew")
        assert edge is not None, "for-each loop variable's type was not indexed"
        assert edge.metadata["cross_file"] is True

    def test_catch_parameter_resolves(self):
        cpg = build({
            "Runner.java": """public class Runner {
                void run(){ try { boom(); } catch (MyError e) { e.report(); } }
                void boom(){}
            }""",
            "MyError.java": "public class MyError { void report(){} }",
        })
        assert find_call(cpg, "run", "report") is not None

    def test_interface_typed_field_resolves_to_interface_method(self):
        """The Spring-style case: a field declared as an interface. Interfaces
        are not CPG class nodes, but their METHODS are, so this must resolve."""
        cpg = build({
            "Controller.java": "public class Controller { private SalesService svc; void go(){ svc.sell(); } }",
            "SalesService.java": "public interface SalesService { void sell(); }",
        })
        edge = find_call(cpg, "go", "sell")
        assert edge is not None, "interface-typed receiver did not resolve"
        assert cpg.nodes[edge.target_id].file_path.endswith("SalesService.java")

    def test_static_call_on_corpus_class_resolves(self):
        cpg = build({
            "A.java": "public class A { void go(){ Util.help(); } }",
            "Util.java": "public class Util { static void help(){} }",
        })
        edge = find_call(cpg, "go", "help")
        assert edge is not None
        assert edge.metadata["resolution"] == "static_class"

    def test_fully_qualified_static_call_uses_last_segment(self):
        cpg = build({
            "A.java": "public class A { void go(){ com.demo.Util.help(); } }",
            "Util.java": "public class Util { static void help(){} }",
        })
        assert find_call(cpg, "go", "help") is not None


class TestSameFileAndHierarchy:
    def test_unqualified_call_resolves_in_same_file(self):
        cpg = build({"A.java": "public class A { void go(){ helper(); } void helper(){} }"})
        edge = find_call(cpg, "go", "helper")
        assert edge is not None
        assert edge.metadata["resolution"] == "same_file"
        assert edge.metadata["cross_file"] is False

    def test_this_qualified_call_resolves(self):
        """`this.m()` is a javalang SELECTOR with qualifier=None, which must not
        be confused with a chained call."""
        cpg = build({"A.java": "public class A { void go(){ this.helper(); } void helper(){} }"})
        edge = find_call(cpg, "go", "helper")
        assert edge is not None
        assert edge.metadata["receiver_kind"] == THIS

    def test_inherited_call_resolves_up_the_chain(self):
        cpg = build({
            "Sub.java": "public class Sub extends Base { void go(){ inherited(); } }",
            "Base.java": "public class Base { void inherited(){} }",
        })
        edge = find_call(cpg, "go", "inherited")
        assert edge is not None
        assert edge.metadata["resolution"] == "inherited"
        assert edge.metadata["cross_file"] is True

    def test_super_call_resolves(self):
        """`super.m()` is a SuperMethodInvocation - a different javalang node
        type that was previously not collected at all."""
        cpg = build({
            "Sub.java": "public class Sub extends Base { void go(){ super.inherited(); } }",
            "Base.java": "public class Base { void inherited(){} }",
        })
        edge = find_call(cpg, "go", "inherited")
        assert edge is not None
        assert edge.metadata["resolution"] == "super_chain"
        assert edge.metadata["receiver_kind"] == SUPER

    def test_field_inherited_from_a_supertype_resolves(self):
        """A protected field on a base class, used from a subclass - idiomatic
        in Spring-style hierarchies, and previously an untrackable receiver."""
        cpg = build({
            "Sub.java": "public class Sub extends Base { void go(){ repo.persist(); } }",
            "Base.java": "public class Base { protected Repo repo; }",
            "Repo.java": "public class Repo { void persist(){} }",
        })
        edge = find_call(cpg, "go", "persist")
        assert edge is not None, "inherited field's declared type was not found"
        assert edge.metadata["cross_file"] is True

    def test_unique_global_name_resolves_for_unqualified_call(self):
        """An unqualified call must target the enclosing type, a supertype or a
        static import, so a globally unique name is the only candidate."""
        cpg = build({
            "A.java": "public class A { void go(){ staticallyImported(); } }",
            "B.java": "public class B { static void staticallyImported(){} }",
        })
        edge = find_call(cpg, "go", "staticallyImported")
        assert edge is not None
        assert edge.metadata["resolution"] == "unique_global"


class TestInheritsEdges:
    def test_class_to_class_inherits_edge_is_created(self):
        cpg = build({
            "Sub.java": "public class Sub extends Base { }",
            "Base.java": "public class Base { }",
        })
        inherits = [e for e in cpg.edges if e.edge_type == "inherits"]
        assert len(inherits) == 1
        assert cpg.nodes[inherits[0].source_id].name == "Sub"
        assert cpg.nodes[inherits[0].target_id].name == "Base"

    def test_interface_supertype_is_counted_not_guessed(self):
        """Interfaces are not class nodes, so there is nothing to point at;
        the gap must be counted rather than silently dropped or mis-attributed."""
        cpg = build({
            "Impl.java": "public class Impl implements Service { public void go(){} }",
            "Service.java": "public interface Service { void go(); }",
        })
        assert not [e for e in cpg.edges if e.edge_type == "inherits"]
        assert cpg.call_resolution_stats["inherits_supertype_not_a_node"] >= 1

    def test_external_supertype_is_counted_separately(self):
        cpg = build({"A.java": "public class A extends RuntimeException { }"})
        assert cpg.call_resolution_stats["inherits_outside_corpus"] >= 1


# --------------------------------------------------------------------------- #
# Deliberate refusals - each must stay a refusal, with its own reason
# --------------------------------------------------------------------------- #

class TestRefusals:
    def test_library_typed_receiver_is_external_type(self):
        cpg = build({
            "A.java": 'public class A { private java.util.List<String> xs; void go(){ xs.add("x"); } }',
        })
        assert find_call(cpg, "go", "add") is None
        assert "external_type" in reasons(cpg)

    def test_library_static_call_is_external_static_type(self):
        cpg = build({"A.java": 'public class A { void go(){ String.format("%s", 1); } }'})
        assert "external_static_type" in reasons(cpg)

    def test_chained_call_is_refused(self):
        cpg = build({
            "A.java": "public class A { private B b; void go(){ getB().hit(); } B getB(){ return b; } }",
            "B.java": "public class B { void hit(){} }",
        })
        # `getB()` itself resolves; `.hit()` off its result does not.
        assert find_call(cpg, "go", "getB") is not None
        assert find_call(cpg, "go", "hit") is None
        assert "chained_call" in reasons(cpg)

    def test_lambda_parameter_is_attributed_to_missing_inference(self):
        """javalang exposes a lambda parameter's name but never its type, so
        this is a documented inference limit - not an indexing oversight."""
        cpg = build({
            "A.java": """public class A {
                void go(java.util.List<Batch> xs){ xs.stream().filter(b -> b.ready()); }
            }""",
            "Batch.java": "public class Batch { boolean ready(){ return true; } }",
        })
        assert find_call(cpg, "go", "ready") is None
        assert "lambda_parameter" in reasons(cpg)

    def test_ambiguous_simple_type_name_is_refused(self):
        cpg = build({
            "A.java": "public class A { private Dup d; void go(){ d.hit(); } }",
            "one/Dup.java": "public class Dup { void hit(){} }",
            "two/Dup.java": "public class Dup { void hit(){} }",
        })
        assert find_call(cpg, "go", "hit") is None
        assert "ambiguous_type" in reasons(cpg)

    def test_dotted_field_chain_is_refused(self):
        cpg = build({
            "A.java": "public class A { private B b; void go(){ b.inner.hit(); } }",
            "B.java": "public class B { public C inner; }",
            "C.java": "public class C { void hit(){} }",
        })
        assert find_call(cpg, "go", "hit") is None
        assert "dotted_qualifier" in reasons(cpg)

    def test_unknown_method_on_known_type_is_refused(self):
        cpg = build({
            "A.java": "public class A { private B b; void go(){ b.notThere(); } }",
            "B.java": "public class B { void hit(){} }",
        })
        assert find_call(cpg, "go", "notThere") is None
        assert "no_such_method" in reasons(cpg)

    def test_ambiguous_unqualified_name_is_refused(self):
        """Two same-named methods, neither in the caller's file nor its chain:
        there is no basis to choose, so no edge."""
        cpg = build({
            "A.java": "public class A { void go(){ shared(); } }",
            "B.java": "public class B { void shared(){} }",
            "C.java": "public class C { void shared(){} }",
        })
        assert find_call(cpg, "go", "shared") is None
        assert "ambiguous_unqualified" in reasons(cpg)

    def test_unique_global_is_never_applied_to_a_qualified_call(self):
        """The dangerous case: `logger.info(...)` must NOT capture a lone
        in-corpus `info()` just because the name happens to be unique."""
        cpg = build({
            "A.java": "public class A { private org.slf4j.Logger logger; void go(){ logger.info(\"x\"); } }",
            "B.java": "public class B { void info(){} }",
        })
        assert find_call(cpg, "go", "info") is None


# --------------------------------------------------------------------------- #
# Statistics and unit-level resolver behaviour
# --------------------------------------------------------------------------- #

class TestStatistics:
    def test_statistics_report_cross_file_and_coverage(self):
        cpg = build({
            "Service.java": "public class Service { private Repo repo; void save(){ repo.persist(); } }",
            "Repo.java": "public class Repo { void persist(){} }",
        })
        stats = cpg.get_statistics()
        assert stats["cross_file_call_edges"] >= 1
        assert stats["call_graph_edges"] >= 1
        assert 0.0 < stats["call_resolution_rate"] <= 1.0
        assert "qualifier_type" in stats["call_resolution_strategies"]

    def test_resolution_rate_is_resolved_over_all_invocations(self):
        cpg = build({
            "A.java": 'public class A { void go(){ helper(); String.format("x"); } void helper(){} }',
        })
        stats = cpg.get_statistics()
        attempted = stats["call_graph_edges"] + stats["unresolved_calls"]
        assert attempted == 2
        assert stats["call_resolution_rate"] == pytest.approx(0.5)

    def test_disabling_call_graph_produces_no_call_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "A.java"), "w", encoding="utf-8") as handle:
                handle.write("public class A { void go(){ helper(); } void helper(){} }")
            cpg = CodePropertyGraph(language="java", include_call_graph=False)
            cpg.build_from_directory(tmp)
        assert not call_edges(cpg)
        assert not cpg.unresolved_calls


class TestCoverageSplit:
    """A low overall rate on a library- and Lombok-heavy corpus is a property
    of the corpus. The split keeps that distinguishable from a resolver gap."""

    def test_out_of_scope_refusals_are_excluded_from_the_in_corpus_rate(self):
        split = coverage({
            "same_file": 6,
            "qualifier_type": 4,
            "external_type": 50,          # library receiver - out of scope
            "generated_accessor": 30,     # Lombok - out of scope
            "chained_call": 20,           # needs type inference - out of scope
            "no_such_method": 10,         # addressable
        })
        assert split["invocations"] == 120
        assert split["resolved"] == 10
        assert split["out_of_scope"] == 100
        assert split["addressable_unresolved"] == 10
        assert split["rate"] == pytest.approx(10 / 120)
        assert split["rate_in_corpus"] == pytest.approx(0.5)

    def test_inherits_bookkeeping_is_not_counted_as_an_invocation(self):
        split = coverage({"same_file": 1, "inherits_outside_corpus": 99})
        assert split["invocations"] == 1
        assert split["rate"] == pytest.approx(1.0)

    def test_empty_tally_does_not_divide_by_zero(self):
        split = coverage({})
        assert split["rate"] == 0.0
        assert split["rate_in_corpus"] == 0.0

    def test_lombok_generated_members_are_attributed_not_blamed(self):
        cpg = build({
            "Svc.java": "public class Svc { private User u; void go(){ u.getName(); } }",
            "User.java": "import lombok.Data;\n@Data\npublic class User { private String name; }",
        })
        assert find_call(cpg, "go", "getName") is None
        assert "generated_accessor" in reasons(cpg)
        # ...and it must NOT inflate the addressable-gap count.
        split = cpg.get_statistics()["call_coverage"]
        assert split["addressable_unresolved"] == 0

    def test_slf4j_log_field_is_attributed_to_generation(self):
        cpg = build({
            "Svc.java": 'import lombok.extern.slf4j.Slf4j;\n@Slf4j\npublic class Svc { void go(){ log.info("x"); } }',
        })
        assert "generated_logger_field" in reasons(cpg)

    def test_accessor_on_a_plain_class_is_still_a_real_miss(self):
        """Without an accessor-generating annotation, a missing getter is a
        genuine no_such_method, not something to excuse."""
        cpg = build({
            "Svc.java": "public class Svc { private User u; void go(){ u.getName(); } }",
            "User.java": "public class User { private String name; }",
        })
        assert "no_such_method" in reasons(cpg)
        assert "generated_accessor" not in reasons(cpg)


class TestSimpleTypeName:
    """javalang splits a qualified type across `sub_type`, so reading `.name`
    alone yields "java" for `java.util.List`."""

    @pytest.mark.parametrize("source,expected", [
        ("private java.util.List<String> x;", "List"),
        ("private Helper x;", "Helper"),
        ("private Map.Entry<String,Integer> x;", "Entry"),
        ("private int x;", "int"),
    ])
    def test_last_link_in_the_chain_wins(self, source, expected):
        import javalang
        tree = javalang.parse.parse("public class T { %s }" % source)
        field = next(node for _p, node in tree.filter(javalang.tree.FieldDeclaration))
        assert simple_type_name(field.type) == expected

    def test_none_type_is_tolerated(self):
        assert simple_type_name(None) is None


class TestResolverUnit:
    """The resolver in isolation, without a CPG - cheaper to pin edge cases."""

    def _resolver(self):
        resolver = CallResolver()
        resolver.index_class("Repo", "Repo.java", [])
        resolver.index_method("method_Repo.java_persist", "persist", "Repo.java")
        resolver.index_class("Service", "Service.java", [])
        resolver.index_method("method_Service.java_save", "save", "Service.java")
        return resolver

    def _call(self, **kwargs):
        defaults = dict(caller_node_id="method_Service.java_save", caller_file="Service.java",
                        caller_method="save", member="persist", kind=QUALIFIED, qualifier="repo")
        defaults.update(kwargs)
        return PendingCall(**defaults)

    def test_resolves_via_indexed_field_type(self):
        resolver = self._resolver()
        resolver.index_field_type("Service.java", "repo", "Repo")
        resolution = resolver.resolve(self._call())
        assert resolution.resolved
        assert resolution.target_node_id == "method_Repo.java_persist"
        assert resolution.strategy == "qualifier_type"

    def test_method_scope_shadows_field_of_the_same_name(self):
        resolver = self._resolver()
        resolver.index_class("Other", "Other.java", [])
        resolver.index_field_type("Service.java", "repo", "Other")
        resolver.index_local_type("Service.java", "save", "repo", "Repo")
        resolution = resolver.resolve(self._call())
        assert resolution.target_node_id == "method_Repo.java_persist"

    def test_supertype_cycle_does_not_hang(self):
        resolver = CallResolver()
        resolver.index_class("A", "A.java", ["B"])
        resolver.index_class("B", "B.java", ["A"])
        resolver.index_method("method_A.java_go", "go", "A.java")
        resolution = resolver.resolve(self._call(
            caller_file="A.java", caller_method="go", member="missing", kind=PLAIN, qualifier=""))
        assert not resolution.resolved

    def test_resolve_all_returns_reasons_and_counts(self):
        resolver = self._resolver()
        resolver.index_field_type("Service.java", "repo", "Repo")
        resolved, unresolved, stats = resolver.resolve_all([
            self._call(),
            self._call(member="nope"),
        ])
        assert len(resolved) == 1
        assert len(unresolved) == 1
        assert unresolved[0]["reason"] == "no_such_method"
        assert stats["qualifier_type"] == 1
        assert stats["no_such_method"] == 1
