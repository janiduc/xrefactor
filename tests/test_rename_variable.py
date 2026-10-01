"""
Tests for the deterministic rename_variable refactoring.

Two kinds of assertion here, and the second matters more than the first:
  * applied renames produce the exact expected source AND compile-verify;
  * every unsafe situation REFUSES with a specific code rather than producing a
    subtly wrong rename. Silent breakage is the only unacceptable outcome.
"""

import javalang
import pytest

from src.refactor import result as R
from src.refactor import verify
from src.refactor.naming import StaticNameProposer
from src.refactor.patterns.rename_variable import rename_local
from src.refactor.verify import verify_refactoring

javac_available = verify.find_javac()[0] is not None
requires_javac = pytest.mark.skipif(not javac_available, reason="javac not on PATH")


def _parse(source):
    return javalang.parse.parse(source)


def _method(source, name="run"):
    tree = _parse(source)
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        if node.name == name:
            return node, tree
    raise AssertionError(f"method {name} not found")


def _rename(source, old, method="run", **kwargs):
    node, tree = _method(source, method)
    return rename_local(source, node, old, file_tree=tree, **kwargs)


class TestSuccessfulRenames:
    def test_renames_local_declaration_and_all_uses(self):
        source = """public class Subject {
    public int run() {
        int tmp = 1;
        tmp = tmp + 2;
        return tmp;
    }
}
"""
        outcome = _rename(source, "tmp", new_name="accumulator")
        assert outcome.applied, outcome.to_dict()
        assert outcome.metadata["occurrences_renamed"] == 4
        assert "int accumulator = 1;" in outcome.modified_source
        assert "accumulator = accumulator + 2;" in outcome.modified_source
        assert "return accumulator;" in outcome.modified_source
        assert "tmp" not in outcome.modified_source

    def test_renames_parameter_including_signature(self):
        source = """public class Subject {
    public int run(int tmp) {
        return tmp * 2;
    }
}
"""
        outcome = _rename(source, "tmp", new_name="factor")
        assert outcome.applied, outcome.to_dict()
        assert "public int run(int factor)" in outcome.modified_source
        assert "return factor * 2;" in outcome.modified_source

    def test_does_not_touch_same_name_in_other_methods(self):
        source = """public class Subject {
    public int run() {
        int tmp = 1;
        return tmp;
    }
    public int other() {
        int tmp = 9;
        return tmp;
    }
}
"""
        outcome = _rename(source, "tmp", new_name="counter")
        assert outcome.applied, outcome.to_dict()
        assert "int counter = 1;" in outcome.modified_source
        # The other method keeps its own local untouched.
        assert "int tmp = 9;" in outcome.modified_source

    def test_skips_field_access_and_method_calls_of_the_same_name(self):
        source = """public class Subject {
    private int value;
    public int run() {
        int tmp = 1;
        this.value = tmp;
        return helper(tmp);
    }
    private int helper(int v) { return v; }
}
"""
        outcome = _rename(source, "tmp", new_name="seed")
        assert outcome.applied, outcome.to_dict()
        assert "this.value = seed;" in outcome.modified_source
        assert "return helper(seed);" in outcome.modified_source

    def test_preserves_comments_and_formatting(self):
        source = """public class Subject {
    public int run() {
        // tmp stays mentioned in this comment
        int    tmp   =   1;   /* and here */
        return tmp;
    }
}
"""
        outcome = _rename(source, "tmp", new_name="seed")
        assert outcome.applied, outcome.to_dict()
        # Comments are untouched (they produce no tokens) and spacing survives.
        assert "// tmp stays mentioned in this comment" in outcome.modified_source
        assert "/* and here */" in outcome.modified_source
        assert "int    seed   =   1;" in outcome.modified_source

    def test_updates_javadoc_param_tag(self):
        source = """public class Subject {
    /**
     * Does work.
     * @param tmp the input
     */
    public int run(int tmp) {
        return tmp;
    }
}
"""
        outcome = _rename(source, "tmp", new_name="input")
        assert outcome.applied, outcome.to_dict()
        assert "@param input the input" in outcome.modified_source
        assert outcome.metadata["javadoc_param_updated"] is True

    def test_proposer_supplies_the_name_and_is_recorded(self):
        source = """public class Subject {
    public int run() {
        int tmp = 1;
        return tmp;
    }
}
"""
        outcome = _rename(source, "tmp", proposer=StaticNameProposer(["itemCount"]))
        assert outcome.applied, outcome.to_dict()
        assert outcome.metadata["new_name"] == "itemCount"
        assert outcome.metadata["name_source"] == "neural"

    def test_falls_back_to_heuristic_when_proposal_is_illegal(self):
        source = """public class Subject {
    public int run() {
        int tmp = 1;
        return tmp;
    }
}
"""
        # "class" is reserved; "run" already exists in the file.
        outcome = _rename(source, "tmp", proposer=StaticNameProposer(["class", "run"]))
        assert outcome.applied, outcome.to_dict()
        assert outcome.metadata["name_source"] == "heuristic"
        reasons = {r["reason"] for r in outcome.metadata["rejected_candidates"]}
        assert "reserved word" in reasons
        assert "identifier already present in this file" in reasons


class TestRefusals:
    def test_refuses_when_name_is_redeclared_in_the_method(self):
        source = """public class Subject {
    public int run() {
        { int tmp = 1; }
        int tmp = 2;
        return tmp;
    }
}
"""
        outcome = _rename(source, "tmp", new_name="seed")
        assert not outcome.applied
        assert outcome.refusal_code == R.SHADOWED_OR_REDECLARED

    def test_refuses_when_declared_inside_a_lambda(self):
        source = """import java.util.function.Supplier;
public class Subject {
    public int run() {
        Supplier<Integer> s = () -> { int tmp = 1; return tmp; };
        return s.get();
    }
}
"""
        outcome = _rename(source, "tmp", new_name="seed")
        assert not outcome.applied
        assert outcome.refusal_code == R.NESTED_SCOPE_DECLARATION

    def test_refuses_when_a_field_shares_the_name_and_scope_is_captured(self):
        source = """import java.util.function.Supplier;
public class Subject {
    private int tmp;
    public int run() {
        int tmp = 1;
        Supplier<Integer> s = () -> tmp;
        return s.get();
    }
}
"""
        outcome = _rename(source, "tmp", new_name="seed")
        assert not outcome.applied
        assert outcome.refusal_code == R.AMBIGUOUS_FIELD_CAPTURE

    def test_refuses_when_the_name_is_a_label(self):
        source = """public class Subject {
    public int run() {
        int tmp = 0;
        tmp:
        for (int i = 0; i < 2; i++) { break tmp; }
        return tmp;
    }
}
"""
        outcome = _rename(source, "tmp", new_name="seed")
        assert not outcome.applied
        assert outcome.refusal_code in (R.NAME_IS_LABEL, R.SHADOWED_OR_REDECLARED)

    def test_refuses_reflectively_bound_parameter(self):
        source = """public class Subject {
    public String run(@PathVariable String id) {
        return id;
    }
}
"""
        outcome = _rename(source, "id", new_name="identifier")
        assert not outcome.applied
        assert outcome.refusal_code == R.REFLECTIVE_PARAM_NAME

    def test_allows_reflective_parameter_when_name_is_explicit(self):
        source = """public class Subject {
    public String run(@PathVariable("id") String id) {
        return id;
    }
}
"""
        outcome = _rename(source, "id", new_name="identifier")
        assert outcome.applied, outcome.to_dict()

    def test_refuses_name_that_is_already_acceptable(self):
        source = """public class Subject {
    public int run() {
        int customerBalance = 1;
        return customerBalance;
    }
}
"""
        outcome = _rename(source, "customerBalance")
        assert not outcome.applied
        assert outcome.refusal_code == R.NAME_ALREADY_ACCEPTABLE

    def test_refuses_unknown_target(self):
        source = """public class Subject {
    public int run() { return 1; }
}
"""
        outcome = _rename(source, "nothingHere", new_name="x2")
        assert not outcome.applied
        assert outcome.refusal_code == R.TARGET_NOT_A_LOCAL

    def test_conventional_loop_counter_is_left_alone(self):
        source = """public class Subject {
    public int run() {
        int total = 0;
        for (int i = 0; i < 3; i++) { total += i; }
        return total;
    }
}
"""
        outcome = _rename(source, "i")
        assert not outcome.applied
        assert outcome.refusal_code == R.NAME_ALREADY_ACCEPTABLE


@requires_javac
class TestRenamesAreCompileVerified:
    """An applied rename must survive differential compilation - this is what
    makes "provably correct" more than a claim."""

    def test_applied_rename_verifies(self):
        source = """public class Subject {
    public int run(int tmp) {
        int val = tmp * 2;
        return val + tmp;
    }
}
"""
        outcome = _rename(source, "val", new_name="doubled")
        assert outcome.applied, outcome.to_dict()

        verdict = verify_refactoring(source, outcome.modified_source,
                                     file_name="Subject.java",
                                     touched_identifiers=outcome.touched_identifiers)
        assert verdict.passed, verdict.to_dict()

    def test_a_deliberately_incomplete_rename_is_caught(self):
        """Mutation check: if the engine ever missed an occurrence, the verifier
        must notice. Here we break it on purpose."""
        source = """public class Subject {
    public int run() {
        int tmp = 1;
        return tmp + tmp;
    }
}
"""
        # Rename only the declaration, leaving the uses dangling.
        broken = source.replace("int tmp = 1;", "int seed = 1;")
        verdict = verify_refactoring(source, broken, file_name="Subject.java",
                                     touched_identifiers=["tmp", "seed"])
        assert verdict.broken, verdict.to_dict()
        assert verdict.verdict == verify.BROKEN_DANGLING_SYMBOL


@pytest.mark.slow
@requires_javac
class TestCorpusProperty:
    """The load-bearing evidence: across real code, every rename the engine
    APPLIES must compile-verify. Refusing is always acceptable; silently
    breaking the program is not."""

    def test_applied_renames_never_break_real_files(self):
        import glob
        import os

        files = sorted(glob.glob(os.path.join("..", "Data", "**", "*.java"), recursive=True))
        assert files, "no corpus files found"

        applied = refused = broken = inconclusive = 0
        refusal_codes = {}
        failures = []
        attempts = 0
        target_attempts = 60

        for path in files:
            if attempts >= target_attempts:
                break
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                source = handle.read()
            try:
                tree = javalang.parse.parse(source)
            except Exception:
                continue

            basename = os.path.basename(path)
            for _p, method in tree.filter(javalang.tree.MethodDeclaration):
                if attempts >= target_attempts or not method.body:
                    break
                # Pick the first local or parameter we can name.
                target = None
                if method.parameters:
                    target = method.parameters[0].name
                else:
                    for _q, decl in method.filter(javalang.tree.LocalVariableDeclaration):
                        if decl.declarators:
                            target = decl.declarators[0].name
                            break
                if not target:
                    continue

                attempts += 1
                outcome = rename_local(source, method, target, file_tree=tree,
                                        new_name=f"xrVerified{attempts}",
                                        enforce_poor_name=False)
                if not outcome.applied:
                    refused += 1
                    refusal_codes[outcome.refusal_code] = refusal_codes.get(outcome.refusal_code, 0) + 1
                    continue

                verdict = verify_refactoring(source, outcome.modified_source,
                                             file_name=basename,
                                             touched_identifiers=outcome.touched_identifiers)
                if verdict.broken:
                    broken += 1
                    failures.append(f"{basename}::{method.name} rename {target} -> "
                                     f"{verdict.verdict}: {verdict.detail}")
                elif verdict.passed:
                    applied += 1
                else:
                    inconclusive += 1

        print(f"\ncorpus rename property: {attempts} attempted | "
              f"{applied} applied+verified | {refused} refused | "
              f"{inconclusive} inconclusive | {broken} BROKEN")
        print(f"refusal codes: {refusal_codes}")
        assert broken == 0, "engine produced broken refactorings:\n" + "\n".join(failures)
        assert applied > 0, "no rename was applied and verified; the test proved nothing"
