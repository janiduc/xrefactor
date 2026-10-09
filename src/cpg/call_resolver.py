"""
Resolving method invocations to the declarations they actually target.

Why this exists: the CPG builder used to synthesise every call edge as
`method_<CALLER's file>_<invoked name>`, i.e. it assumed every invocation
targeted a method in the file doing the invoking. On this corpus that made
93% of call edges point at node ids that do not exist (they were silently
materialised as phantom nodes by NetworkX) and produced exactly ZERO
cross-file call edges - in a framework whose whole premise is cross-file
reasoning. Downstream, `dead_code` became 95% of all detected smells because
almost nothing had a tracked caller, and `high_coupling` could never fire at
all because a resolvable target always shared the caller's file by
construction.

Java method resolution properly requires full type resolution, which needs
the dependency classpath these repos are not built with. So this module does
NOT attempt to be a type checker. It follows the same discipline as
`src/refactor/`: resolve only when a specific, stated rule applies, and
record an explicit reason when it does not. An unresolved call produces NO
edge rather than a guessed one, and every unresolved call is counted and
attributed, so the call graph's real coverage is visible instead of implied.

Resolution strategies, strongest first (see `CallResolver.resolve`):

    super_chain      `super.m()`  -> a method named m in a supertype's file
    same_file        `m()` / `this.m()` -> a method named m in the same file
    inherited        `m()` / `this.m()` -> a method named m up the extends chain
    unique_global    `m()` where exactly ONE method named m exists anywhere
    static_class     `Foo.m()` where Foo is a class declared in this corpus
    qualifier_type   `x.m()` where x's DECLARED type is a class in this corpus

Deliberately NOT resolved (each is a counted refusal, not a silent drop):

    external_type       `x.m()` where x's type is not declared in this corpus
                        (`List.add`, `String.trim`, ...) - a library call, and
                        correctly absent from an intra-corpus call graph.
    lambda_parameter    `xs.stream().filter(b -> b.m())` - javalang exposes the
                        parameter's NAME but never its type (it comes from the
                        functional interface), so resolving it would require
                        generic/stream type inference.
    unknown_qualifier   `x.m()` where x is not a field/param/local we tracked.
    dotted_qualifier    `a.b.m()` - would need the type of `a.b`, which needs
                        real field-type resolution through another class.
    chained_call        `getX().m()` - the receiver is an expression whose type
                        we cannot infer without a type checker.
    ambiguous_unqualified  `m()` with several same-named methods and no
                        same-file or inherited candidate to disambiguate.
    ambiguous_type      the qualifier's type name is declared by more than one
                        class in this corpus (same simple name, two packages).
    no_such_method      the receiving class was identified but declares no
                        method of that name (typically an inherited method
                        whose supertype is outside this corpus).

Known approximations, stated rather than hidden:

  * File granularity. Method node ids are keyed by file, so "the method named
    m in class T" is implemented as "the method named m in T's file". Accurate
    under the one-public-class-per-file convention these repos follow; a file
    with several top-level classes conflates their members.
  * `unique_global` is the one rule that reaches outside the caller's own type
    hierarchy. It is sound for UNQUALIFIED calls only - an unqualified call
    must target the enclosing type, a supertype, or a static import, so if
    exactly one method of that name exists in the corpus it is the only
    candidate. It is never applied to qualified calls, where a unique name
    would otherwise wrongly capture library calls.
  * Overloads collapse. The builder keys methods by name alone, so
    `m(int)` and `m(String)` share one node; resolution picks that single node
    and arity is not checked.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Receiver kinds, determined by the builder from javalang's AST shape.
PLAIN = "plain"          # m()
THIS = "this"            # this.m()
SUPER = "super"          # super.m()
QUALIFIED = "qualified"  # x.m() / Foo.m()
CHAINED = "chained"      # getX().m() - receiver is an expression

# How deep to walk an extends/implements chain before giving up. Real
# hierarchies in this corpus are shallow; the bound also guards against a
# cycle that a malformed or partially-parsed file could introduce.
_MAX_SUPERTYPE_DEPTH = 8

# Lombok annotations that SYNTHESISE members at compile time. The generated
# accessors and logger field do not exist in the source javalang parses, so
# calls to them are unresolvable for any source-only analysis. They are
# attributed to code generation rather than counted as resolver failures,
# because the distinction decides whether a low resolution rate is a bug to
# fix or a property of the corpus - on Lombok-heavy repos it is the latter.
_ACCESSOR_GENERATING_ANNOTATIONS = frozenset({
    "Data", "Getter", "Setter", "Value", "Builder", "SuperBuilder",
    "AllArgsConstructor", "NoArgsConstructor", "RequiredArgsConstructor",
    "EqualsAndHashCode", "ToString", "Accessors",
})
_LOGGER_GENERATING_ANNOTATIONS = frozenset({
    "Slf4j", "XSlf4j", "Log", "Log4j", "Log4j2", "CommonsLog", "JBossLog", "Flogger",
})
_GENERATED_LOGGER_NAMES = frozenset({"log", "logger", "LOG", "LOGGER"})
_ACCESSOR_NAME_RE = re.compile(r"^(get|set|is)[A-Z0-9_]")


# Refusals that NO source-only resolver can fix, because the target is not in
# the parsed source at all (a library method, or a member Lombok synthesises at
# compile time) or because deciding it needs real type inference (the type of an
# expression, a lambda parameter, or a field reached through another field).
# Separated from the addressable refusals so the headline coverage number can be
# read correctly: a low overall rate on a Lombok-heavy, library-heavy corpus is a
# property of the corpus, while an addressable refusal is a resolver gap.
OUT_OF_SCOPE_REASONS = frozenset({
    "external_type", "external_static_type",
    "generated_accessor", "generated_logger_field",
    "chained_call", "lambda_parameter", "dotted_qualifier",
})

# Strategy labels that mean "an edge was produced". Resolution strategies gain a
# `_inherited` suffix when the method was found up the receiver's chain.
def is_resolved_strategy(strategy: str) -> bool:
    return strategy in {
        "same_file", "inherited", "unique_global", "super_chain",
        "static_class", "static_class_inherited",
        "qualifier_type", "qualifier_type_inherited",
    }


def coverage(stats: Dict[str, int]) -> Dict[str, Any]:
    """Split a strategy tally into resolved / out-of-scope / addressable.

    `rate` is resolved over every invocation seen - the honest headline, but it
    is bounded well below 1.0 by library and generated code. `rate_in_corpus`
    is resolved over only those invocations a source-only resolver could ever
    decide, which is the number that reflects resolver quality. Both are
    reported, never one alone.
    """
    resolved = sum(count for strategy, count in stats.items()
                   if is_resolved_strategy(strategy))
    out_of_scope = sum(count for strategy, count in stats.items()
                       if strategy in OUT_OF_SCOPE_REASONS)
    # Inheritance bookkeeping is counted in the same tally but is not a call.
    call_total = sum(count for strategy, count in stats.items()
                     if not strategy.startswith("inherits_"))
    addressable = call_total - resolved - out_of_scope
    return {
        "invocations": call_total,
        "resolved": resolved,
        "out_of_scope": out_of_scope,
        "addressable_unresolved": addressable,
        "rate": (resolved / call_total) if call_total else 0.0,
        "rate_in_corpus": (resolved / (resolved + addressable)) if (resolved + addressable) else 0.0,
    }


@dataclass
class PendingCall:
    """One invocation, captured during the first pass before the global index
    of every declaration in the corpus is complete."""

    caller_node_id: str
    caller_file: str
    caller_method: str
    member: str
    kind: str
    qualifier: Optional[str] = None
    line: int = 0


@dataclass
class CallResolution:
    """Either a resolved target node id plus the rule that found it, or a
    reason why no edge should be created."""

    target_node_id: Optional[str]
    strategy: str

    @property
    def resolved(self) -> bool:
        return self.target_node_id is not None


@dataclass
class CallResolver:
    """Global declaration index for one corpus, built during the CPG's first
    pass and queried during its second.

    Every mapping is keyed by SIMPLE name, because that is what an invocation
    site gives us; ambiguity between two same-named classes is detected and
    refused rather than resolved arbitrarily.
    """

    # simple method name -> method node ids, across every file
    methods_by_name: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    # simple class name -> the files declaring it (>1 means ambiguous)
    files_by_class: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    # file -> simple names of its classes' supertypes (extends + implements)
    supertypes_by_file: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    # (file, field name) -> declared simple type name
    field_types: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # (file, method name, local/param name) -> declared simple type name
    local_types: Dict[Tuple[str, str, str], str] = field(default_factory=dict)
    # (file, method name, name) for bindings whose NAME is known but whose type
    # is not written down anywhere - currently lambda parameters, whose type
    # comes from the functional interface being implemented. Tracked so an
    # unresolvable receiver can be attributed to inference we do not do, rather
    # than reported as a name the indexer failed to see.
    untyped_names: Set[Tuple[str, str, str]] = field(default_factory=set)
    # file -> simple names of the annotations on its type declarations, used
    # only to attribute a miss to compile-time code generation (see
    # _ACCESSOR_GENERATING_ANNOTATIONS).
    annotations_by_file: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # node_id -> file, for turning an indexed method back into its file
    _file_of_node: Dict[str, str] = field(default_factory=dict)

    # --------------------------------------------------------------- indexing

    def index_method(self, node_id: str, name: str, file_path: str) -> None:
        if node_id not in self._file_of_node:
            # Overloads share a node id (see module docstring); index it once.
            self.methods_by_name[name].append(node_id)
        self._file_of_node[node_id] = file_path

    def index_class(self, name: str, file_path: str, supertypes: List[str]) -> None:
        if file_path not in self.files_by_class[name]:
            self.files_by_class[name].append(file_path)
        for supertype in supertypes:
            if supertype and supertype not in self.supertypes_by_file[file_path]:
                self.supertypes_by_file[file_path].append(supertype)

    def index_field_type(self, file_path: str, name: str, type_name: Optional[str]) -> None:
        if type_name:
            self.field_types[(file_path, name)] = type_name

    def index_local_type(self, file_path: str, method_name: str, name: str,
                         type_name: Optional[str]) -> None:
        if type_name:
            self.local_types[(file_path, method_name, name)] = type_name

    def index_untyped_name(self, file_path: str, method_name: str, name: str) -> None:
        self.untyped_names.add((file_path, method_name, name))

    def index_type_annotations(self, file_path: str, annotation_names: Set[str]) -> None:
        self.annotations_by_file[file_path].update(annotation_names)

    # ---------------------------------------------------------------- lookups

    def _method_in_file(self, name: str, file_path: str) -> Optional[str]:
        for node_id in self.methods_by_name.get(name, ()):
            if self._file_of_node.get(node_id) == file_path:
                return node_id
        return None

    def _supertype_files(self, file_path: str) -> List[str]:
        """Files of every supertype reachable from `file_path`, breadth-first.

        Supertypes declared outside this corpus simply contribute nothing,
        which is why a call inherited from a framework base class stays
        unresolved rather than being misattributed.
        """
        seen: Set[str] = {file_path}
        ordered: List[str] = []
        frontier = [(file_path, 0)]
        while frontier:
            current_file, depth = frontier.pop(0)
            if depth >= _MAX_SUPERTYPE_DEPTH:
                continue
            for supertype in self.supertypes_by_file.get(current_file, ()):
                for candidate_file in self.files_by_class.get(supertype, ()):
                    if candidate_file in seen:
                        continue
                    seen.add(candidate_file)
                    ordered.append(candidate_file)
                    frontier.append((candidate_file, depth + 1))
        return ordered

    def _method_up_chain(self, name: str, file_path: str) -> Optional[str]:
        for supertype_file in self._supertype_files(file_path):
            node_id = self._method_in_file(name, supertype_file)
            if node_id is not None:
                return node_id
        return None

    def _declared_type_of(self, call: PendingCall, receiver: str) -> Optional[str]:
        """Declared simple type of a receiver name, in Java's own lookup order:
        method scope (a local legitimately shadows a field), then this file's
        fields, then fields inherited from a supertype.

        The inherited step matters in these codebases: a `protected` field on a
        base class used from a subclass is idiomatic (Spring's
        `BaseController`-style hierarchies), and without it every such receiver
        was reported as an untrackable name.
        """
        local = self.local_types.get((call.caller_file, call.caller_method, receiver))
        if local is not None:
            return local
        own_field = self.field_types.get((call.caller_file, receiver))
        if own_field is not None:
            return own_field
        for supertype_file in self._supertype_files(call.caller_file):
            inherited_field = self.field_types.get((supertype_file, receiver))
            if inherited_field is not None:
                return inherited_field
        return None

    # -------------------------------------------------------------- resolution

    def resolve(self, call: PendingCall) -> CallResolution:
        """Apply the strategy ladder in the module docstring to one invocation."""
        if call.kind == SUPER:
            node_id = self._method_up_chain(call.member, call.caller_file)
            return (CallResolution(node_id, "super_chain") if node_id
                    else CallResolution(None, "no_such_method"))

        if call.kind in (PLAIN, THIS):
            return self._resolve_unqualified(call)

        if call.kind == QUALIFIED:
            return self._resolve_qualified(call)

        # CHAINED, or anything the builder could not classify.
        return CallResolution(None, "chained_call")

    def _resolve_unqualified(self, call: PendingCall) -> CallResolution:
        """`m()` or `this.m()`: the target must be in the enclosing type, a
        supertype, or a static import - so same-file, then up the chain, then
        a globally unique name are each the only remaining candidate."""
        node_id = self._method_in_file(call.member, call.caller_file)
        if node_id is not None:
            return CallResolution(node_id, "same_file")

        node_id = self._method_up_chain(call.member, call.caller_file)
        if node_id is not None:
            return CallResolution(node_id, "inherited")

        candidates = self.methods_by_name.get(call.member, ())
        if len(candidates) == 1:
            return CallResolution(candidates[0], "unique_global")
        if not candidates:
            return CallResolution(None, "no_such_method")
        return CallResolution(None, "ambiguous_unqualified")

    def _resolve_qualified(self, call: PendingCall) -> CallResolution:
        """`x.m()` or `Foo.m()`: find the receiver's type, then look for the
        method in that type's file (and up its chain)."""
        qualifier = (call.qualifier or "").strip()
        if not qualifier:
            return CallResolution(None, "unknown_qualifier")

        segments = qualifier.split(".")

        # `com.y.Other.stat()` / `Foo.stat()`: a static call names its class
        # directly, so the LAST segment is the type.
        type_name = segments[-1]
        if type_name in self.files_by_class:
            return self._resolve_in_type(type_name, call, "static_class")

        # `x.m()`: a single-segment receiver we can look up a declared type for.
        if len(segments) > 1:
            # `a.b.m()` needs the type of `a.b`, i.e. real field resolution
            # through another class - out of scope, so refuse.
            return CallResolution(None, "dotted_qualifier")

        declared_type = self._declared_type_of(call, qualifier)
        if declared_type is None:
            if (qualifier in _GENERATED_LOGGER_NAMES
                    and self.annotations_by_file.get(call.caller_file, ())
                    and set(self.annotations_by_file[call.caller_file])
                    & _LOGGER_GENERATING_ANNOTATIONS):
                # Lombok's @Slf4j and friends synthesise the `log` field, so it
                # has no declaration for us to read a type from.
                return CallResolution(None, "generated_logger_field")
            if (call.caller_file, call.caller_method, qualifier) in self.untyped_names:
                # A lambda parameter: we know the name but its type comes from
                # the functional interface / stream element type, which needs
                # generic inference this resolver does not attempt.
                return CallResolution(None, "lambda_parameter")
            if qualifier[:1].isupper():
                # `String.format(...)`, `Collectors.toList()`: a static call on a
                # type this corpus does not declare. Distinguished from a plain
                # unknown receiver by the Java convention that type names are
                # capitalised - a heuristic, but it separates "library static
                # call" (expected, uninteresting) from "a name we failed to
                # track" (a resolver gap worth looking at).
                return CallResolution(None, "external_static_type")
            return CallResolution(None, "unknown_qualifier")
        if declared_type not in self.files_by_class:
            # A library/JDK type - correctly outside an intra-corpus call graph.
            return CallResolution(None, "external_type")
        return self._resolve_in_type(declared_type, call, "qualifier_type")

    def _resolve_in_type(self, type_name: str, call: PendingCall,
                         strategy: str) -> CallResolution:
        files = self.files_by_class.get(type_name, ())
        if len(files) != 1:
            # Two classes share this simple name; picking one would be a guess.
            return CallResolution(None, "ambiguous_type")
        target_file = files[0]
        node_id = self._method_in_file(call.member, target_file)
        if node_id is not None:
            return CallResolution(node_id, strategy)
        node_id = self._method_up_chain(call.member, target_file)
        if node_id is not None:
            return CallResolution(node_id, strategy + "_inherited")
        if (_ACCESSOR_NAME_RE.match(call.member)
                and set(self.annotations_by_file.get(target_file, ()))
                & _ACCESSOR_GENERATING_ANNOTATIONS):
            # The receiving type is a Lombok @Data/@Getter class, so its
            # accessors are generated at compile time and simply are not in the
            # source text. Not a resolver gap.
            return CallResolution(None, "generated_accessor")
        return CallResolution(None, "no_such_method")

    # ------------------------------------------------------------------ driver

    def resolve_all(self, calls: List[PendingCall]
                    ) -> Tuple[List[Tuple[PendingCall, CallResolution]], List[dict], Counter]:
        """Resolve a whole batch.

        Returns (resolved pairs, unresolved records, strategy counts) so the
        caller can build edges from the first and REPORT the second rather
        than discarding it - the point of this module is that the call graph's
        coverage is measurable.
        """
        resolved: List[Tuple[PendingCall, CallResolution]] = []
        unresolved: List[dict] = []
        stats: Counter = Counter()
        for call in calls:
            resolution = self.resolve(call)
            stats[resolution.strategy] += 1
            if resolution.resolved:
                resolved.append((call, resolution))
            else:
                unresolved.append({
                    "caller": call.caller_node_id,
                    "member": call.member,
                    "qualifier": call.qualifier,
                    "kind": call.kind,
                    "line": call.line,
                    "reason": resolution.strategy,
                })
        return resolved, unresolved, stats


def simple_type_name(type_node: Any) -> Optional[str]:
    """Simple name of a javalang type reference.

    javalang splits a qualified type across `sub_type`: `java.util.List` parses
    as name='java' with sub_type chain util -> List, so reading `.name` alone
    yields "java". The last link in the chain is the simple name ("List"), and
    for a nested type (`Map.Entry`) it is the nested class ("Entry") - which is
    the name an invocation site would match against.
    """
    if type_node is None:
        return None
    name = getattr(type_node, "name", None)
    current = getattr(type_node, "sub_type", None)
    while current is not None:
        candidate = getattr(current, "name", None)
        if candidate:
            name = candidate
        current = getattr(current, "sub_type", None)
    return name
