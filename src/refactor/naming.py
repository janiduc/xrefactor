"""
Choosing a replacement identifier - the hybrid seam.

The neural model is good at *suggesting* a plausible name and bad at applying an
edit correctly; the AST engine is the reverse. So a `NameProposer` proposes
ordered candidates and the engine validates and applies the first legal one.
`name_source` records which actually won, so the neural contribution is
measurable rather than assumed.

Candidate validation is deliberately over-strict in one respect: a candidate is
rejected if the identifier appears ANYWHERE in the file. Without a classpath we
cannot resolve fields, inherited members, static imports or nested classes, so
"absent from the whole file" is the cheapest sound way to guarantee the rename
introduces no collision. javac's `already.defined` is the backstop.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

from javalang.tokenizer import Identifier, Keyword

from src.refactor.offsets import tokenize_cached

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_LOWER_CAMEL_RE = re.compile(r"^[a-z][A-Za-z0-9_$]*$")

# Reserved beyond javalang's Keyword set (contextual keywords and literals).
_EXTRA_RESERVED = {
    "true", "false", "null", "var", "record", "yield", "sealed", "permits",
    "exports", "module", "requires", "opens", "provides", "uses", "to", "with",
    "transitive",
}

# Names that are poor enough to be worth renaming. Anything else is left alone,
# so the engine does not churn already-readable code.
_GENERIC_NAME_RE = re.compile(
    r"^(tmp|temp|val|value|data|obj|o|a|b|foo|bar|str|s|flag|result|res|ret|x1|x2)\d*$",
    re.IGNORECASE,
)
_CONVENTIONAL_SHORT = {"i", "j", "k", "n", "e", "_"}

_COLLECTION_TYPES = {"List", "ArrayList", "LinkedList", "Set", "HashSet", "TreeSet",
                     "Collection", "Queue", "Deque", "Iterable"}
_MAP_TYPES = {"Map", "HashMap", "TreeMap", "LinkedHashMap", "ConcurrentHashMap"}

_GETTER_PREFIXES = ("get", "find", "load", "build", "create", "to", "parse", "read", "fetch")


@dataclass
class RenameContext:
    """What a proposer gets to reason about."""

    old_name: str
    kind: str  # "local" | "parameter"
    declared_type_text: str = ""
    initializer_source: str = ""
    method_source: str = ""
    enclosing_class_name: str = ""
    taken_names: Sequence[str] = field(default_factory=tuple)


class NameProposer(Protocol):
    def propose(self, ctx: RenameContext) -> List[str]:
        """Ordered candidate names, best first."""
        ...


def _lower_camel(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", text or "")
    if not text:
        return ""
    return text[0].lower() + text[1:]


def _simple_type_name(type_text: str) -> str:
    """`java.util.List<String>[]` -> `List`."""
    if not type_text:
        return ""
    head = type_text.split("<", 1)[0].strip()
    head = head.replace("[", " ").replace("]", " ").strip()
    head = head.split()[-1] if head.split() else ""
    return head.split(".")[-1]


def _type_arguments(type_text: str) -> List[str]:
    """Top-level generic arguments of `Map<String, List<Integer>>`."""
    if "<" not in type_text:
        return []
    inner = type_text[type_text.index("<") + 1:type_text.rindex(">")] if ">" in type_text else ""
    args, depth, current = [], 0, ""
    for ch in inner:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())
    return args


def _pluralise(word: str) -> str:
    if not word:
        return ""
    if word.endswith("s"):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


class HeuristicNameProposer:
    """Deterministic fallback proposer. Derives a name from the declared type or
    the initializer, which is what a human reviewer would do."""

    def propose(self, ctx: RenameContext) -> List[str]:
        candidates: List[str] = []
        type_text = ctx.declared_type_text or ""
        simple = _simple_type_name(type_text)
        initializer = (ctx.initializer_source or "").strip()

        is_array = "[]" in type_text
        if simple in _MAP_TYPES:
            args = _type_arguments(type_text)
            if len(args) == 2:
                key, value = _simple_type_name(args[0]), _simple_type_name(args[1])
                if key and value:
                    candidates.append(_lower_camel(value) + "By" + key[0].upper() + key[1:])
            candidates.append(_lower_camel(simple))
        elif simple in _COLLECTION_TYPES or is_array:
            args = _type_arguments(type_text)
            element = _simple_type_name(args[0]) if args else _simple_type_name(type_text)
            if element:
                candidates.append(_pluralise(_lower_camel(element)))
                candidates.append(_lower_camel(element) + "List")

        # From a getter-ish initializer: `getCustomer()` -> `customer`
        call = re.match(r"^(?:[\w.]+\.)?(\w+)\s*\(", initializer)
        if call:
            name = call.group(1)
            for prefix in _GETTER_PREFIXES:
                if name.lower().startswith(prefix) and len(name) > len(prefix):
                    candidates.append(_lower_camel(name[len(prefix):]))
                    break

        # From `new Foo(...)`
        creator = re.match(r"^new\s+([\w.]+)", initializer)
        if creator:
            candidates.append(_lower_camel(_simple_type_name(creator.group(1))))

        if type_text.strip() == "boolean":
            if call:
                candidates.append("is" + (call.group(1)[0].upper() + call.group(1)[1:]))
            candidates.append("isEnabled")

        if simple and simple not in _MAP_TYPES and simple not in _COLLECTION_TYPES:
            candidates.append(_lower_camel(simple))

        # Last resort, always syntactically valid.
        candidates.append("renamed" + ctx.old_name[0].upper() + ctx.old_name[1:])
        for suffix in range(2, 6):
            candidates.append(f"renamed{ctx.old_name[0].upper()}{ctx.old_name[1:]}{suffix}")

        seen, ordered = set(), []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered


class StaticNameProposer:
    """Proposes a fixed list - used when a caller (or the neural model) has
    already decided on candidate names."""

    def __init__(self, names: Sequence[str]):
        self._names = list(names)

    def propose(self, ctx: RenameContext) -> List[str]:
        return list(self._names)


def identifiers_in_source(source: str) -> set:
    """Every Identifier token in the file. Used for the collision check."""
    tokens = tokenize_cached(source)
    if tokens is None:
        return set()
    return {t.value for t in tokens if isinstance(t, Identifier)}


def validate_candidate(candidate: str,
                       old_name: str,
                       file_source: str,
                       existing_identifiers: Optional[set] = None) -> Tuple[bool, str]:
    """Is `candidate` a legal, non-colliding replacement for `old_name`?"""
    if not candidate:
        return False, "empty"
    if not _IDENTIFIER_RE.match(candidate):
        return False, "not a valid Java identifier"
    if len(candidate) > 64:
        return False, "longer than 64 characters"
    if candidate in Keyword.VALUES or candidate in _EXTRA_RESERVED:
        return False, "reserved word"
    if candidate == old_name:
        return False, "same as the current name"
    if not _LOWER_CAMEL_RE.match(candidate):
        return False, "not lowerCamelCase"
    identifiers = existing_identifiers if existing_identifiers is not None \
        else identifiers_in_source(file_source)
    if candidate in identifiers:
        # Over-strict on purpose: without a classpath this is the cheapest sound
        # guarantee that no field, inherited member or static import collides.
        return False, "identifier already present in this file"
    return True, ""


def name_is_poor(name: str) -> bool:
    """Only rename names that are actually worth renaming."""
    if not name:
        return False
    if name in _CONVENTIONAL_SHORT:
        return False  # idiomatic loop counters are fine
    if len(name) <= 2:
        return True
    if _GENERIC_NAME_RE.match(name):
        return True
    if "_" in name and not name.isupper():
        return True  # my_var
    if name.isupper() and len(name) > 1:
        return True  # MYVAR for a local
    return False


def choose_name(ctx: RenameContext,
                file_source: str,
                proposer: Optional[NameProposer] = None) -> Tuple[Optional[str], str, List[dict]]:
    """Return (chosen_name, name_source, rejected_candidates).

    Tries the given proposer first (the neural seam), then the deterministic
    heuristic, so a failed neural suggestion degrades rather than blocking.
    """
    existing = identifiers_in_source(file_source)
    rejected: List[dict] = []

    attempts: List[Tuple[str, NameProposer]] = []
    if proposer is not None:
        attempts.append(("neural", proposer))
    attempts.append(("heuristic", HeuristicNameProposer()))

    for source_label, candidate_proposer in attempts:
        try:
            candidates = candidate_proposer.propose(ctx)
        except Exception as e:
            rejected.append({"name": None, "reason": f"{source_label} proposer failed: {e}"})
            continue
        for candidate in candidates:
            ok, reason = validate_candidate(candidate, ctx.old_name, file_source, existing)
            if ok:
                return candidate, source_label, rejected
            rejected.append({"name": candidate, "reason": reason, "source": source_label})

    return None, "none", rejected
