"""
The result type every refactoring pattern returns, plus refusal codes.

Design stance: a pattern either produces a transformation it can justify, or it
REFUSES with a specific code. It never guesses. A high refusal rate is an
honest outcome and is reported as such (see evaluate.py's `verified_yield`,
which divides by candidates rather than by attempts, so refusing most cases
cannot be mistaken for accuracy).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Refusal codes
# --------------------------------------------------------------------------- #

# Shared
NO_PARSE = "NO_PARSE"
NO_EXTENT = "NO_EXTENT"
NODE_NOT_FOUND = "NODE_NOT_FOUND"
TOKEN_AST_MISMATCH = "TOKEN_AST_MISMATCH"
SYNTHESIS_FAILED = "SYNTHESIS_FAILED"
PATTERN_NOT_SUPPORTED = "PATTERN_NOT_SUPPORTED"

# rename_variable / improve_naming
SHADOWED_OR_REDECLARED = "SHADOWED_OR_REDECLARED"
NESTED_SCOPE_DECLARATION = "NESTED_SCOPE_DECLARATION"
AMBIGUOUS_FIELD_CAPTURE = "AMBIGUOUS_FIELD_CAPTURE"
NAME_IS_LABEL = "NAME_IS_LABEL"
NAME_COLLIDES_WITH_MEMBER = "NAME_COLLIDES_WITH_MEMBER"
NAME_IS_TYPE = "NAME_IS_TYPE"
REFLECTIVE_PARAM_NAME = "REFLECTIVE_PARAM_NAME"
FIELD_RENAME_CROSS_FILE = "FIELD_RENAME_CROSS_FILE"
METHOD_RENAME_CROSS_FILE = "METHOD_RENAME_CROSS_FILE"
NAME_ALREADY_ACCEPTABLE = "NAME_ALREADY_ACCEPTABLE"
NO_VALID_NEW_NAME = "NO_VALID_NEW_NAME"
TARGET_NOT_A_LOCAL = "TARGET_NOT_A_LOCAL"

# remove_dead_code
NOT_PRIVATE = "NOT_PRIVATE"
HAS_ANNOTATIONS = "HAS_ANNOTATIONS"
REFLECTION_HOOK = "REFLECTION_HOOK"
STILL_REFERENCED = "STILL_REFERENCED"
NOTHING_UNREACHABLE = "NOTHING_UNREACHABLE"

# simplify_condition
NO_SIMPLIFIABLE_CONDITION = "NO_SIMPLIFIABLE_CONDITION"

# extract_method
CONSTRUCTOR_UNSUPPORTED = "CONSTRUCTOR_UNSUPPORTED"
NO_BODY = "NO_BODY"
RANGE_NOT_TOP_LEVEL = "RANGE_NOT_TOP_LEVEL"
RANGE_TOO_SMALL = "RANGE_TOO_SMALL"
RANGE_IS_WHOLE_BODY = "RANGE_IS_WHOLE_BODY"
ESCAPE_RETURN = "ESCAPE_RETURN"
ESCAPE_BREAK = "ESCAPE_BREAK"
ESCAPE_CONTINUE = "ESCAPE_CONTINUE"
WRITES_TO_OUTER_LOCAL = "WRITES_TO_OUTER_LOCAL"
MULTIPLE_OUTPUTS = "MULTIPLE_OUTPUTS"
OUTPUT_IS_FINAL = "OUTPUT_IS_FINAL"
TYPE_IS_VAR = "TYPE_IS_VAR"
C_STYLE_ARRAY_DECLARATOR = "C_STYLE_ARRAY_DECLARATOR"
SHADOWED_IN_RANGE = "SHADOWED_IN_RANGE"
USES_METHOD_TYPE_PARAMETER = "USES_METHOD_TYPE_PARAMETER"
CONTAINS_LOCAL_TYPE_DECLARATION = "CONTAINS_LOCAL_TYPE_DECLARATION"
PARAM_COUNT_EXCEEDS_LIMIT = "PARAM_COUNT_EXCEEDS_LIMIT"
NAME_COLLISION = "NAME_COLLISION"


@dataclass
class RefactorResult:
    """Outcome of attempting one deterministic refactoring."""

    applied: bool
    pattern: str
    refusal_code: Optional[str] = None
    detail: str = ""
    # Whole FILE after the edit - what the verifier compiles.
    modified_source: Optional[str] = None
    # The refactored member's own text - what a suggestion shows the user.
    refactored_member: Optional[str] = None
    original_member: Optional[str] = None
    # Names introduced or removed; the verifier uses these to tell a dangling
    # reference apart from an ordinary missing-classpath error.
    touched_identifiers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def refuse(cls, pattern: str, code: str, detail: str = "") -> "RefactorResult":
        return cls(applied=False, pattern=pattern, refusal_code=code, detail=detail)

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "pattern": self.pattern,
            "refusal_code": self.refusal_code,
            "detail": self.detail,
            "touched_identifiers": self.touched_identifiers,
            "metadata": self.metadata,
        }
