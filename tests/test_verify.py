"""
Tests for src/refactor/verify.py.

The cases that matter most are the ones that must NOT pass: a verifier that
reports success when javac never ran, or when the baseline is untrustworthy,
would make every downstream "provably correct" claim worthless.
"""

import pytest

from src.refactor import verify
from src.refactor.verify import (
    BROKEN_DANGLING_SYMBOL,
    BROKEN_DOES_NOT_PARSE,
    BROKEN_STRUCTURAL,
    INCONCLUSIVE_BASELINE_PARSE_FAIL,
    INCONCLUSIVE_JAVAC_MISSING,
    PASSED_COMPILES_CLEAN,
    PASSED_NO_NEW_STRUCTURAL,
    RESOLUTION,
    STRUCTURAL,
    TYPE,
    UNKNOWN,
    classify_key,
    parse_diagnostics,
    tokens_balanced,
    verify_refactoring,
)

javac_available = verify.find_javac()[0] is not None
requires_javac = pytest.mark.skipif(not javac_available, reason="javac not on PATH")

CLEAN = """public class Subject {
    public int add(int a, int b) {
        int sum = a + b;
        return sum;
    }
}
"""

# Realistic: references a type that cannot resolve without the project classpath.
WITH_UNRESOLVED_IMPORT = """import org.nonexistent.Thing;
public class Subject {
    public Thing make() {
        return new Thing();
    }
    public int add(int a, int b) {
        int sum = a + b;
        return sum;
    }
}
"""


class TestClassification:
    def test_syntax_error_is_structural(self):
        assert classify_key("compiler.err.expected") == STRUCTURAL
        assert classify_key("compiler.err.premature.eof") == STRUCTURAL

    def test_scope_and_dataflow_errors_are_structural(self):
        # These are exactly the keys that catch engine bugs.
        assert classify_key("compiler.err.already.defined") == STRUCTURAL
        assert classify_key("compiler.err.var.might.not.have.been.initialized") == STRUCTURAL
        assert classify_key("compiler.err.unreachable.stmt") == STRUCTURAL
        assert classify_key("compiler.err.missing.ret.stmt") == STRUCTURAL

    def test_missing_classpath_errors_are_resolution(self):
        assert classify_key("compiler.err.doesnt.exist") == RESOLUTION
        assert classify_key("compiler.err.cant.resolve.location") == RESOLUTION
        assert classify_key("compiler.err.not.def.access.class.intf.cant.access") == RESOLUTION

    def test_classpath_cascades_that_look_structural_are_only_type(self):
        """Measured on unmodified corpus files, so they must not fail a refactoring."""
        assert classify_key("compiler.err.ref.ambiguous") == TYPE
        assert classify_key("compiler.err.method.does.not.override.superclass") == TYPE

    def test_unknown_key_is_not_silently_treated_as_fine(self):
        assert classify_key("compiler.err.some.key.invented.in.jdk.99") == UNKNOWN


class TestDiagnosticParsing:
    def test_parses_raw_diagnostic_line(self):
        stderr = "Broken.java:2:48: compiler.err.expected: ';'\n1 error\n"
        diags, unparsed = parse_diagnostics(stderr)
        assert len(diags) == 1
        assert diags[0].key == "compiler.err.expected"
        assert diags[0].args == "';'"
        assert diags[0].line == 2 and diags[0].column == 48
        assert unparsed == []  # the "1 error" footer is not an error

    def test_fingerprint_ignores_position(self):
        stderr = ("A.java:2:4: compiler.err.expected: ';'\n"
                  "A.java:99:7: compiler.err.expected: ';'\n")
        diags, _ = parse_diagnostics(stderr)
        assert diags[0].fingerprint() == diags[1].fingerprint()

    def test_warnings_and_notes_are_not_errors(self):
        stderr = ("A.java:1:1: compiler.warn.prob.found.req: x\n"
                  "- compiler.note.unchecked.filename: A.java\n")
        diags, unparsed = parse_diagnostics(stderr)
        assert diags == []

    def test_mentions_finds_touched_identifier_in_args(self):
        stderr = "A.java:4:44: compiler.err.cant.resolve.location: kindname.class, oldName, , ,\n"
        diags, _ = parse_diagnostics(stderr)
        assert diags[0].mentions(["oldName"])
        assert not diags[0].mentions(["unrelated"])


class TestPreGates:
    def test_tokens_balanced_detects_mangled_splice(self):
        assert tokens_balanced("class A { void f() { } }")
        assert not tokens_balanced("class A { void f() { }")
        assert not tokens_balanced("class A { void f() ) }")

    def test_braces_in_literals_do_not_unbalance(self):
        source = 'class A { void f() { String s = "}}}"; char c = ' + chr(39) + "}" + chr(39) + "; } }"
        assert tokens_balanced(source)

    def test_unparseable_original_is_inconclusive_not_a_pass(self):
        result = verify_refactoring("class A { this is not java", CLEAN)
        assert result.verdict == INCONCLUSIVE_BASELINE_PARSE_FAIL
        assert not result.passed
        assert result.inconclusive


@requires_javac
class TestDifferentialVerification:
    def test_identical_source_passes(self):
        result = verify_refactoring(CLEAN, CLEAN)
        assert result.passed, result.to_dict()

    def test_clean_compile_is_reported_as_such(self):
        result = verify_refactoring(CLEAN, CLEAN)
        assert result.verdict == PASSED_COMPILES_CLEAN

    def test_baseline_resolution_errors_are_ignored(self):
        """The whole point of the differential protocol: a file that cannot
        resolve its imports must still be verifiable."""
        modified = WITH_UNRESOLVED_IMPORT.replace("int sum = a + b;", "int total = a + b;") \
                                          .replace("return sum;", "return total;")
        result = verify_refactoring(WITH_UNRESOLVED_IMPORT, modified)
        assert result.passed, result.to_dict()
        assert result.baseline_diagnostic_count > 0, "fixture should have baseline errors"
        assert result.verdict == PASSED_NO_NEW_STRUCTURAL

    def test_deleted_semicolon_is_broken_structural(self):
        modified = CLEAN.replace("int sum = a + b;", "int sum = a + b")
        result = verify_refactoring(CLEAN, modified)
        assert result.verdict == BROKEN_DOES_NOT_PARSE or result.verdict == BROKEN_STRUCTURAL
        assert result.broken

    def test_structural_error_that_still_parses_is_caught_by_javac(self):
        """A duplicate local declaration parses fine but javac rejects it -
        this is the class of bug the AST engine could introduce."""
        modified = CLEAN.replace("int sum = a + b;", "int sum = a + b; int sum = 2;")
        result = verify_refactoring(CLEAN, modified)
        assert result.verdict == BROKEN_STRUCTURAL, result.to_dict()
        assert any("already.defined" in d.key for d in result.new_structural)

    def test_missed_rename_occurrence_is_dangling_symbol(self):
        """Renaming the declaration but not a use leaves a dangling reference.
        Without the touched-identifier rule this would look like just another
        ignorable resolution error."""
        modified = CLEAN.replace("int sum = a + b;", "int renamedSum = a + b;")
        result = verify_refactoring(CLEAN, modified,
                                     touched_identifiers=["sum", "renamedSum"])
        assert result.verdict == BROKEN_DANGLING_SYMBOL, result.to_dict()
        assert "sum" in result.dangling_identifiers

    def test_unrelated_resolution_error_is_not_dangling(self):
        """Same shape as above but the new unresolved name is NOT one we touched,
        so it must stay an ignored resolution error."""
        modified = CLEAN.replace("return sum;", "return sum + Helper.offset();")
        result = verify_refactoring(CLEAN, modified, touched_identifiers=["sum"])
        assert result.verdict != BROKEN_DANGLING_SYMBOL, result.to_dict()

    def test_no_class_files_are_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        verify_refactoring(CLEAN, CLEAN)
        assert list(tmp_path.rglob("*.class")) == []

    def test_completes_quickly(self):
        verify.clear_baseline_cache()
        result = verify_refactoring(CLEAN, CLEAN)
        assert result.elapsed_seconds < 15, f"took {result.elapsed_seconds}s"


class TestNoFalsePasses:
    def test_missing_javac_is_inconclusive_never_a_pass(self, monkeypatch):
        monkeypatch.setattr(verify, "_javac_checked", True)
        monkeypatch.setattr(verify, "_javac_path", None)
        monkeypatch.setattr(verify, "_javac_version", None)

        result = verify_refactoring(CLEAN, CLEAN)
        assert result.verdict == INCONCLUSIVE_JAVAC_MISSING
        assert not result.passed
        assert result.inconclusive

    def test_inconclusive_is_neither_passed_nor_broken(self):
        result = verify.VerificationResult(verdict=verify.INCONCLUSIVE_TYPE_DELTA)
        assert not result.passed and not result.broken and result.inconclusive

    def test_environment_info_records_the_compiler(self):
        info = verify.environment_info()
        assert "javac_version" in info and "javac_available" in info
