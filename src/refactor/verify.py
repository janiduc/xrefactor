"""
Compile-verification for refactored Java, via a DIFFERENTIAL javac protocol.

Why differential rather than "does it compile": these repos are compiled without
their dependency classpath, so javac legitimately reports thousands of
resolution errors on completely untouched code (measured on this corpus: 3376
`cant.resolve.location`, 1267 `doesnt.exist`). Worse, some diagnostics that look
structural are actually classpath cascades (`ref.ambiguous`,
`method.does.not.override.superclass`), and `already.defined` was observed on an
*unmodified* file. No whitelist or blacklist survives that. So we compile the
ORIGINAL and the MODIFIED file and compare diagnostic multisets: only what the
refactoring newly introduced counts.

Diagnostics are requested with `-XDrawDiagnostics`, which emits stable
`compiler.err.*` keys instead of localised prose, one per line:

    Broken.java:2:48: compiler.err.expected: ';'
    Unresolved.java:4:44: compiler.err.already.defined: kindname.variable, y, ...

`-XDshould-stop.ifNoError=FLOW` stops after flow analysis, so type-checking and
definite-assignment errors are still reported but no `.class` files are written
(verified).

Three buckets:
  RESOLUTION  - ignored (missing classpath), EXCEPT when the diagnostic's
                arguments name an identifier the refactoring touched, which
                means we broke a reference -> BROKEN_DANGLING_SYMBOL.
  TYPE        - inconclusive: can indicate a real bug (wrong argument order in
                an extraction) or just a cascade shift, so neither pass nor fail.
  STRUCTURAL  - syntax, declaration structure, scope and definite-assignment
                errors. New ones mean the refactoring is BROKEN.

"Compiles" is NOT "correct": a compile pass says we did not break the program's
structure, not that behaviour is preserved. Callers must not conflate the two.
"""

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import javalang
from loguru import logger

from src.refactor.offsets import tokenize_cached

# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #

PASSED_COMPILES_CLEAN = "PASSED_COMPILES_CLEAN"
PASSED_NO_NEW_STRUCTURAL = "PASSED_NO_NEW_STRUCTURAL"
BROKEN_STRUCTURAL = "BROKEN_STRUCTURAL"
BROKEN_DANGLING_SYMBOL = "BROKEN_DANGLING_SYMBOL"
BROKEN_DOES_NOT_PARSE = "BROKEN_DOES_NOT_PARSE"
INCONCLUSIVE_TYPE_DELTA = "INCONCLUSIVE_TYPE_DELTA"
INCONCLUSIVE_UNKNOWN_DIAGNOSTIC = "INCONCLUSIVE_UNKNOWN_DIAGNOSTIC"
INCONCLUSIVE_BASELINE_PARSE_FAIL = "INCONCLUSIVE_BASELINE_PARSE_FAIL"
INCONCLUSIVE_BASELINE_MISMATCH = "INCONCLUSIVE_BASELINE_MISMATCH"
INCONCLUSIVE_JAVAC_MISSING = "INCONCLUSIVE_JAVAC_MISSING"
INCONCLUSIVE_TIMEOUT = "INCONCLUSIVE_TIMEOUT"
VERIFIER_ERROR = "VERIFIER_ERROR"

PASSING_VERDICTS = frozenset({PASSED_COMPILES_CLEAN, PASSED_NO_NEW_STRUCTURAL})
BROKEN_VERDICTS = frozenset({BROKEN_STRUCTURAL, BROKEN_DANGLING_SYMBOL, BROKEN_DOES_NOT_PARSE})

# --------------------------------------------------------------------------- #
# Diagnostic key buckets. Prefix entries ending in '*' match by prefix.
# --------------------------------------------------------------------------- #

_RESOLUTION_KEYS = {
    "doesnt.exist", "cant.resolve", "cant.resolve.args", "cant.resolve.args.params",
    "cant.resolve.location", "cant.resolve.location.args", "cant.resolve.location.args.params",
    "cant.access", "cant.deref", "type.var.cant.be.deref", "package.not.visible",
    "package.empty.or.not.found", "report.access", "module.not.found",
    "import.module.not.found", "no.superclass", "encl.class.required",
    "no.encl.instance.of.type.in.scope", "not.encl.class", "is.preview",
    "cant.read.file", "error.reading.file",
}
_RESOLUTION_PREFIXES = ("not.def.access", "not.def.public", "proc.", "preview.feature.disabled",
                        "feature.not.supported.in.source", "import.module.does.not.read")

_TYPE_KEYS = {
    "prob.found.req", "types.incompatible", "inconvertible.types", "incomparable.types",
    "unexpected.type", "ref.ambiguous", "method.does.not.override.superclass",
    "anonymous.diamond.method.does.not.override.superclass", "does.not.override.abstract",
    "abstract.cant.be.instantiated", "concrete.inheritance.conflict", "cant.inherit.diff.arg",
    "cyclic.inheritance", "wrong.number.type.args", "type.doesnt.take.params",
    "foreach.not.applicable.to.type", "array.req.but.found", "generic.array.creation",
    "except.never.thrown.in.try", "except.already.caught", "const.expr.req",
    "string.const.req", "cant.infer.local.var.type",
}
_TYPE_PREFIXES = ("operator.cant.be.applied", "cant.apply.symbol", "cant.apply.diamond",
                  "override.", "signature.doesnt.match", "name.clash.same.erasure",
                  "unreported.exception", "not.exhaustive")

_STRUCTURAL_KEYS = {
    # syntax
    "expected", "expected1", "expected2", "expected3", "expected4",
    "illegal.start.of.expr", "illegal.start.of.stmt", "illegal.start.of.type", "not.stmt",
    "premature.eof", "unclosed.comment", "unclosed.str.lit", "unclosed.char.lit",
    "unclosed.text.block", "unmatched.quote", "empty.char.lit", "illegal.char",
    "illegal.esc.char", "illegal.underscore", "illegal.unicode.esc", "illegal.dot",
    "dot.class.expected", "orphaned", "else.without.if", "catch.without.try",
    "finally.without.try", "try.without.catch.finally.or.resource.decls",
    "extraneous.semicolon", "illegal.parenthesized.expression",
    "invalid.lambda.parameter.declaration", "unexpected.lambda", "unexpected.mref",
    "invalid.mref", "array.dimension.missing", "malformed.fp.lit", "invalid.hex.number",
    "invalid.binary.number", "illegal.digit.in.octal.literal",
    "illegal.digit.in.binary.literal", "int.number.too.large", "fp.number.too.large",
    "fp.number.too.small", "varargs.and.old.array.syntax", "varargs.must.be.last",
    # declaration structure
    "class.not.allowed", "missing.meth.body.or.decl.abstract", "abstract.meth.cant.have.body",
    "intf.meth.cant.have.body", "native.meth.cant.have.body",
    "invalid.meth.decl.ret.type.req", "repeated.modifier", "illegal.combination.of.modifiers",
    "mod.not.allowed.here", "modifier.not.allowed.here", "duplicate.class",
    "class.public.should.be.in.file", "icls.cant.have.static.decl",
    "static.declaration.not.allowed.in.inner.classes", "local.enum",
    "initializer.not.allowed", "bad.initializer", "void.not.allowed.here",
    "variable.not.allowed",
    # scope / definite assignment - these catch engine bugs
    "already.defined", "already.defined.in.clinit", "already.defined.this.unit",
    "already.defined.single.import", "label.already.in.use", "undef.label",
    "not.loop.label", "break.outside.switch.loop", "break.outside.switch.expression",
    "cont.outside.loop", "continue.outside.switch.expression", "ret.outside.meth",
    "missing.ret.stmt", "unreachable.stmt", "var.might.not.have.been.initialized",
    "var.might.already.be.assigned", "var.might.be.assigned.in.loop",
    "final.parameter.may.not.be.assigned", "cant.assign.val.to.var",
    "cant.assign.val.to.this", "try.resource.may.not.be.assigned",
    "multicatch.parameter.may.not.be.assigned", "cant.ref.non.effectively.final.var",
    "try.with.resources.expr.effectively.final.var", "illegal.forward.ref",
    "illegal.self.ref", "recursive.ctor.invocation", "call.must.only.appear.in.ctor",
    "cant.ref.before.ctor.called", "duplicate.case.label", "duplicate.default.label",
    "default.label.not.allowed", "switch.case.unexpected.statement", "no.switch.expression",
    "invalid.yield", "enum.as.identifier", "assert.as.identifier", "this.as.identifier",
    "underscore.as.identifier", "name.reserved.for.internal.use",
}
_STRUCTURAL_PREFIXES = ("use.of.underscore.not.allowed", "limit.")

RESOLUTION = "RESOLUTION"
TYPE = "TYPE"
STRUCTURAL = "STRUCTURAL"
UNKNOWN = "UNKNOWN"

_DIAG_RE = re.compile(
    r"^(?P<file>[^:]+\.java):(?P<line>\d+)(?::(?P<col>\d+))?: "
    r"(?P<key>compiler\.(?:err|warn|note)\.[A-Za-z0-9_.$]+)(?::\s*(?P<args>.*))?$"
)
_COUNT_FOOTER_RE = re.compile(r"^\d+ (?:error|warning)s?$")

JAVAC_TIMEOUT_SECONDS = 30


def classify_key(key: str) -> str:
    """Bucket a raw `compiler.err.*` key."""
    short = key.split("compiler.err.", 1)[-1] if key.startswith("compiler.err.") else key
    if short in _STRUCTURAL_KEYS or short.startswith(_STRUCTURAL_PREFIXES):
        return STRUCTURAL
    if short in _RESOLUTION_KEYS or short.startswith(_RESOLUTION_PREFIXES):
        return RESOLUTION
    if short in _TYPE_KEYS or short.startswith(_TYPE_PREFIXES):
        return TYPE
    return UNKNOWN


@dataclass(frozen=True)
class Diagnostic:
    key: str
    args: str
    line: int
    column: int

    @property
    def bucket(self) -> str:
        return classify_key(self.key)

    def fingerprint(self) -> Tuple[str, str]:
        """Identity for multiset diffing: line/column are deliberately excluded,
        because any refactoring shifts the lines below it."""
        return (self.key, " ".join(self.args.split()))

    def mentions(self, identifiers: Iterable[str]) -> bool:
        for name in identifiers:
            if name and re.search(rf"\b{re.escape(name)}\b", self.args):
                return True
        return False


@dataclass
class VerificationResult:
    verdict: str
    new_structural: List[Diagnostic] = field(default_factory=list)
    new_type: List[Diagnostic] = field(default_factory=list)
    new_resolution: List[Diagnostic] = field(default_factory=list)
    new_unknown: List[Diagnostic] = field(default_factory=list)
    dangling_identifiers: List[str] = field(default_factory=list)
    baseline_diagnostic_count: int = 0
    detail: str = ""
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.verdict in PASSING_VERDICTS

    @property
    def broken(self) -> bool:
        return self.verdict in BROKEN_VERDICTS

    @property
    def inconclusive(self) -> bool:
        return not self.passed and not self.broken

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "broken": self.broken,
            "inconclusive": self.inconclusive,
            "detail": self.detail,
            "baseline_diagnostic_count": self.baseline_diagnostic_count,
            "new_structural": [f"{d.key}: {d.args}" for d in self.new_structural],
            "new_type": [f"{d.key}: {d.args}" for d in self.new_type],
            "new_unknown": [f"{d.key}: {d.args}" for d in self.new_unknown],
            "dangling_identifiers": self.dangling_identifiers,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


# --------------------------------------------------------------------------- #
# javac invocation
# --------------------------------------------------------------------------- #

_javac_path: Optional[str] = None
_javac_version: Optional[str] = None
_javac_checked = False
_baseline_cache: Dict[str, Tuple[int, Tuple[Diagnostic, ...]]] = {}


def find_javac() -> Tuple[Optional[str], Optional[str]]:
    """(path, version) for javac, resolved once and cached.

    Resolved to an absolute path deliberately: more than one JDK may be
    installed, and a run must not silently switch compilers halfway.
    """
    global _javac_path, _javac_version, _javac_checked
    if _javac_checked:
        return _javac_path, _javac_version
    _javac_checked = True
    path = shutil.which("javac")
    if path is None:
        logger.warning("javac not found on PATH - compile verification is unavailable")
        return None, None
    try:
        proc = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=30)
        _javac_version = (proc.stdout or proc.stderr).strip().splitlines()[0]
    except Exception as e:
        logger.warning(f"javac found at {path} but -version failed: {e}")
        return None, None
    _javac_path = path
    return _javac_path, _javac_version


def _javac_env() -> dict:
    """A deterministic environment: CLASSPATH would make results depend on the
    caller's shell, and JAVA_TOOL_OPTIONS prints a banner to stderr that would
    corrupt diagnostic parsing."""
    env = dict(os.environ)
    env.pop("CLASSPATH", None)
    env.pop("JAVA_TOOL_OPTIONS", None)
    return env


def parse_diagnostics(stderr: str) -> Tuple[List[Diagnostic], List[str]]:
    """Parse `-XDrawDiagnostics` output. Returns (diagnostics, unparsed_lines)."""
    diagnostics: List[Diagnostic] = []
    unparsed: List[str] = []
    for raw_line in stderr.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if _COUNT_FOOTER_RE.match(line.strip()):
            continue
        if line.startswith("- compiler.note."):  # e.g. unchecked/deprecation notes
            continue
        match = _DIAG_RE.match(line.strip())
        if match is None:
            unparsed.append(line.strip())
            continue
        if not match.group("key").startswith("compiler.err."):
            continue  # warnings/notes are not failures
        diagnostics.append(Diagnostic(
            key=match.group("key"),
            args=match.group("args") or "",
            line=int(match.group("line")),
            column=int(match.group("col") or 0),
        ))
    return diagnostics, unparsed


def _compile(source: str, basename: str, stop_after: str = "FLOW") -> Tuple[int, List[Diagnostic], List[str], str]:
    """Compile one source string in isolation.

    Returns (returncode, diagnostics, unparsed_stderr_lines, error_message).
    `error_message` is non-empty only when javac could not be run at all.
    """
    javac, _ = find_javac()
    if javac is None:
        return -1, [], [], "javac missing"

    with tempfile.TemporaryDirectory(prefix="xrefactor_verify_") as tmp:
        src_path = os.path.join(tmp, basename)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        with open(src_path, "w", encoding="utf-8", errors="surrogateescape") as handle:
            handle.write(source)

        cmd = [
            javac,
            "-proc:none", "-nowarn", "-Xlint:none", "-Xmaxwarns", "0",
            "-implicit:none", "-XDrawDiagnostics",
            f"-XDshould-stop.ifNoError={stop_after}",
            "-Xmaxerrs", "10000", "-encoding", "UTF-8",
            "-classpath", "", "-sourcepath", "",
            "-d", out_dir, src_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=JAVAC_TIMEOUT_SECONDS, env=_javac_env())
        except subprocess.TimeoutExpired:
            return -1, [], [], "timeout"
        except Exception as e:
            return -1, [], [], f"javac invocation failed: {e}"

    diagnostics, unparsed = parse_diagnostics(proc.stderr or "")
    # A bare `error: ...` with no file prefix means javac itself failed (bad flag,
    # missing file) - never treat that as a clean compile.
    fatal = [u for u in unparsed if u.startswith("error:")]
    if fatal:
        return proc.returncode, diagnostics, unparsed, f"javac error: {fatal[0]}"
    return proc.returncode, diagnostics, unparsed, ""


def _compile_baseline(source: str, basename: str) -> Tuple[int, Tuple[Diagnostic, ...], str]:
    """Baseline compile, cached by content hash (one file yields many candidate
    refactorings, and recompiling it each time would dominate runtime)."""
    digest = hashlib.sha256(source.encode("utf-8", errors="surrogateescape")).hexdigest()
    if digest in _baseline_cache:
        rc, diags = _baseline_cache[digest]
        return rc, diags, ""
    rc, diags, _unparsed, error = _compile(source, basename)
    if error:
        return rc, tuple(diags), error
    _baseline_cache[digest] = (rc, tuple(diags))
    return rc, tuple(diags), ""


# --------------------------------------------------------------------------- #
# Pre-gates
# --------------------------------------------------------------------------- #

def parses(source: str) -> bool:
    try:
        javalang.parse.parse(source)
        return True
    except Exception:
        return False


def tokens_balanced(source: str) -> bool:
    """Brace/paren/bracket balance over the token stream (cheap, catches a
    mangled splice before a JVM is ever started)."""
    tokens = tokenize_cached(source)
    if tokens is None:
        return False
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack: List[str] = []
    for tok in tokens:
        value = tok.value
        if value in pairs:
            stack.append(value)
        elif value in pairs.values():
            if not stack or pairs[stack.pop()] != value:
                return False
    return not stack


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def verify_refactoring(original_source: str,
                       modified_source: str,
                       file_name: str = "Subject.java",
                       touched_identifiers: Sequence[str] = ()) -> VerificationResult:
    """Differentially verify that `modified_source` did not break `original_source`.

    `touched_identifiers` should list every name the refactoring introduced or
    removed (old name, new name, extracted method name). A NEW resolution error
    naming one of them is not a missing-classpath artefact - it means a
    reference was left dangling, so it is reported as BROKEN_DANGLING_SYMBOL.
    """
    started = time.time()

    javac, _version = find_javac()
    if javac is None:
        return VerificationResult(verdict=INCONCLUSIVE_JAVAC_MISSING,
                                   detail="javac not on PATH",
                                   elapsed_seconds=time.time() - started)

    # Pre-gates: if the original cannot be parsed we have no trustworthy
    # baseline; if only the modified one cannot, the refactoring broke it.
    original_parses = parses(original_source)
    if not original_parses:
        return VerificationResult(verdict=INCONCLUSIVE_BASELINE_PARSE_FAIL,
                                   detail="original source does not parse (javalang)",
                                   elapsed_seconds=time.time() - started)
    if not parses(modified_source) or not tokens_balanced(modified_source):
        return VerificationResult(verdict=BROKEN_DOES_NOT_PARSE,
                                   detail="modified source does not parse or is unbalanced",
                                   elapsed_seconds=time.time() - started)

    base_rc, base_diags, base_error = _compile_baseline(original_source, file_name)
    if base_error:
        verdict = INCONCLUSIVE_TIMEOUT if base_error == "timeout" else VERIFIER_ERROR
        return VerificationResult(verdict=verdict,
                                   detail=f"baseline compile failed: {base_error}",
                                   elapsed_seconds=time.time() - started)

    mod_rc, mod_diags, _unparsed, mod_error = _compile(modified_source, file_name)
    if mod_error:
        verdict = INCONCLUSIVE_TIMEOUT if mod_error == "timeout" else VERIFIER_ERROR
        return VerificationResult(verdict=verdict,
                                   detail=f"modified compile failed: {mod_error}",
                                   baseline_diagnostic_count=len(base_diags),
                                   elapsed_seconds=time.time() - started)

    # Multiset difference: only diagnostics the refactoring ADDED.
    baseline_counts: Dict[Tuple[str, str], int] = {}
    for diag in base_diags:
        key = diag.fingerprint()
        baseline_counts[key] = baseline_counts.get(key, 0) + 1

    new_diags: List[Diagnostic] = []
    for diag in mod_diags:
        key = diag.fingerprint()
        if baseline_counts.get(key, 0) > 0:
            baseline_counts[key] -= 1
        else:
            new_diags.append(diag)

    result = VerificationResult(
        verdict=PASSED_NO_NEW_STRUCTURAL,
        baseline_diagnostic_count=len(base_diags),
        elapsed_seconds=time.time() - started,
    )

    for diag in new_diags:
        bucket = diag.bucket
        if bucket == RESOLUTION:
            if diag.mentions(touched_identifiers):
                result.dangling_identifiers.extend(
                    name for name in touched_identifiers if diag.mentions([name])
                )
                result.new_structural.append(diag)
            else:
                result.new_resolution.append(diag)
        elif bucket == STRUCTURAL:
            result.new_structural.append(diag)
        elif bucket == TYPE:
            result.new_type.append(diag)
        else:
            result.new_unknown.append(diag)

    # A pass requires a baseline that actually ran. If the baseline failed but the
    # modified file compiles clean, something is inconsistent - a refactoring
    # cannot fix unresolved imports - so refuse to call it a pass.
    if base_rc != 0 and mod_rc == 0 and not base_diags:
        result.verdict = INCONCLUSIVE_BASELINE_MISMATCH
        result.detail = "baseline failed without diagnostics while modified compiled clean"
    elif result.dangling_identifiers:
        result.verdict = BROKEN_DANGLING_SYMBOL
        result.detail = (f"new unresolved reference(s) to "
                         f"{sorted(set(result.dangling_identifiers))}")
    elif result.new_structural:
        result.verdict = BROKEN_STRUCTURAL
        result.detail = f"{len(result.new_structural)} new structural error(s): " \
                        f"{result.new_structural[0].key}"
    elif result.new_unknown:
        result.verdict = INCONCLUSIVE_UNKNOWN_DIAGNOSTIC
        result.detail = f"unclassified diagnostic {result.new_unknown[0].key}"
        logger.debug(f"unclassified javac key (extend the bucket lists): "
                     f"{result.new_unknown[0].key}")
    elif result.new_type:
        result.verdict = INCONCLUSIVE_TYPE_DELTA
        result.detail = f"{len(result.new_type)} new type diagnostic(s): {result.new_type[0].key}"
    elif mod_rc == 0 and not mod_diags:
        result.verdict = PASSED_COMPILES_CLEAN
        result.detail = "modified source compiles with no diagnostics at all"
    else:
        result.detail = (f"no new structural errors; "
                         f"{len(result.new_resolution)} new resolution error(s) ignored "
                         f"(no project classpath)")

    result.elapsed_seconds = time.time() - started
    return result


def environment_info() -> dict:
    """Recorded in evaluation reports so numbers are attributable to a compiler."""
    path, version = find_javac()
    return {
        "javac_path": path,
        "javac_version": version,
        "javac_available": path is not None,
        "javac_timeout_seconds": JAVAC_TIMEOUT_SECONDS,
    }


def clear_baseline_cache() -> None:
    _baseline_cache.clear()
