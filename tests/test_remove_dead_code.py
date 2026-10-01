"""
Tests for deterministic remove_dead_code.

The refusals matter more than the deletions here: deleting a method that is
actually invoked by a framework is destructive and unrecoverable from the user's
point of view. The `private`-only gate is the structural reason this engine
cannot make that mistake.
"""

import javalang
import pytest

from src.refactor import result as R
from src.refactor import verify
from src.refactor.patterns.remove_dead_code import (
    remove_dead_code,
    remove_dead_method,
    remove_unreachable_statements,
)
from src.refactor.verify import verify_refactoring

javac_available = verify.find_javac()[0] is not None
requires_javac = pytest.mark.skipif(not javac_available, reason="javac not on PATH")


def _method(source, name):
    tree = javalang.parse.parse(source)
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        if node.name == name:
            return node, tree
    raise AssertionError(f"method {name} not found")


class TestDeletesTrulyDeadPrivateMethods:
    def test_deletes_unreferenced_private_method(self):
        source = """public class Subject {
    public int run() {
        return 1;
    }

    private int neverCalled() {
        return 42;
    }
}
"""
        node, tree = _method(source, "neverCalled")
        outcome = remove_dead_method(source, node, tree)
        assert outcome.applied, outcome.to_dict()
        assert "neverCalled" not in outcome.modified_source
        assert "public int run()" in outcome.modified_source
        assert outcome.metadata["removed"] == "method"

    def test_removes_the_javadoc_with_the_method(self):
        source = """public class Subject {
    public int run() { return 1; }

    /**
     * Describes a method that is about to disappear.
     */
    private int neverCalled() {
        return 42;
    }
}
"""
        node, tree = _method(source, "neverCalled")
        outcome = remove_dead_method(source, node, tree)
        assert outcome.applied, outcome.to_dict()
        assert "about to disappear" not in outcome.modified_source, \
            "javadoc would be orphaned, describing nothing"


class TestRefusalsProtectFrameworkCode:
    def test_refuses_public_method(self):
        source = """public class Subject {
    public int unusedButPublic() { return 1; }
}
"""
        node, tree = _method(source, "unusedButPublic")
        outcome = remove_dead_method(source, node, tree)
        assert not outcome.applied
        assert outcome.refusal_code == R.NOT_PRIVATE

    def test_refuses_protected_method(self):
        source = """public class Subject {
    protected void onCreate() { }
}
"""
        node, tree = _method(source, "onCreate")
        outcome = remove_dead_method(source, node, tree)
        assert not outcome.applied
        assert outcome.refusal_code == R.NOT_PRIVATE

    def test_refuses_annotated_private_method(self):
        """A private @Test or @Bean method is invoked reflectively."""
        source = """public class Subject {
    @Test
    private void shouldWork() { }
}
"""
        node, tree = _method(source, "shouldWork")
        outcome = remove_dead_method(source, node, tree)
        assert not outcome.applied
        assert outcome.refusal_code == R.HAS_ANNOTATIONS

    def test_refuses_serialization_hook(self):
        source = """public class Subject {
    private void writeObject(java.io.ObjectOutputStream out) { }
}
"""
        node, tree = _method(source, "writeObject")
        outcome = remove_dead_method(source, node, tree)
        assert not outcome.applied
        assert outcome.refusal_code == R.REFLECTION_HOOK

    def test_refuses_when_still_called(self):
        source = """public class Subject {
    public int run() { return helper(); }
    private int helper() { return 2; }
}
"""
        node, tree = _method(source, "helper")
        outcome = remove_dead_method(source, node, tree)
        assert not outcome.applied
        assert outcome.refusal_code == R.STILL_REFERENCED

    def test_refuses_when_name_appears_in_a_string_literal(self):
        """Could be reflective lookup by name, so stay out of it."""
        source = """public class Subject {
    public void run() { invokeByName("helper"); }
    private int helper() { return 2; }
    private void invokeByName(String n) { }
}
"""
        node, tree = _method(source, "helper")
        outcome = remove_dead_method(source, node, tree)
        assert not outcome.applied
        assert outcome.refusal_code == R.STILL_REFERENCED

    def test_spring_handler_is_structurally_out_of_reach(self):
        """The documented false-positive class: the CPG sees no caller for a
        request handler, but it is public, so deletion cannot happen."""
        source = """public class Controller {
    @RequestMapping("/x")
    public String handle() { return "ok"; }
}
"""
        node, tree = _method(source, "handle")
        outcome = remove_dead_method(source, node, tree)
        assert not outcome.applied
        assert outcome.refusal_code == R.NOT_PRIVATE


class TestUnreachableStatements:
    def test_removes_statements_after_return(self):
        source = """public class Subject {
    public int run() {
        int a = 1;
        return a;
        int b = 2;
    }
}
"""
        node, _tree = _method(source, "run")
        outcome = remove_unreachable_statements(source, node)
        assert outcome.applied, outcome.to_dict()
        assert "int b = 2;" not in outcome.modified_source
        assert "return a;" in outcome.modified_source
        assert outcome.metadata["statements_removed"] == 1

    def test_refuses_when_nothing_is_unreachable(self):
        source = """public class Subject {
    public int run() {
        int a = 1;
        return a;
    }
}
"""
        node, _tree = _method(source, "run")
        outcome = remove_unreachable_statements(source, node)
        assert not outcome.applied
        assert outcome.refusal_code == R.NOTHING_UNREACHABLE

    def test_dispatcher_prefers_the_local_fix(self):
        source = """public class Subject {
    public int run() { return helper(); }
    private int helper() {
        return 1;
        int dead = 2;
    }
}
"""
        node, tree = _method(source, "helper")
        outcome = remove_dead_code(source, node, tree)
        assert outcome.applied, outcome.to_dict()
        # helper() is still called, so only the unreachable statement goes.
        assert outcome.metadata["removed"] == "unreachable_statements"
        assert "private int helper()" in outcome.modified_source


@requires_javac
class TestDeletionsAreCompileVerified:
    def test_deleting_an_unused_private_method_verifies(self):
        source = """public class Subject {
    public int run() {
        return 1;
    }

    private int neverCalled() {
        return 42;
    }
}
"""
        node, tree = _method(source, "neverCalled")
        outcome = remove_dead_method(source, node, tree)
        assert outcome.applied

        verdict = verify_refactoring(source, outcome.modified_source,
                                     file_name="Subject.java",
                                     touched_identifiers=outcome.touched_identifiers)
        assert verdict.passed, verdict.to_dict()

    def test_deleting_a_called_method_would_be_caught(self):
        """Mutation check: if the STILL_REFERENCED gate ever regressed, the
        verifier must catch the resulting dangling call."""
        source = """public class Subject {
    public int run() { return helper(); }
    private int helper() { return 2; }
}
"""
        broken = source.replace("    private int helper() { return 2; }\n", "")
        verdict = verify_refactoring(source, broken, file_name="Subject.java",
                                     touched_identifiers=["helper"])
        assert verdict.broken, verdict.to_dict()
        assert verdict.verdict == verify.BROKEN_DANGLING_SYMBOL
