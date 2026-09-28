"""
Maps RefactoringMiner's fine-grained refactoring type names to XRefactor's
coarse 10-class taxonomy (see RefactoringGenerator.refactoring_types in
src/transformer/code_generator.py, which this mirrors exactly).

This mapping is a deliberate simplification: RefactoringMiner detects ~90
specific refactoring kinds, but our classifier only reasons about 10 broad
categories. Matching is substring-based (case-insensitive) so it stays robust
to minor naming differences across RefactoringMiner versions. Types that
don't map to any category (e.g. annotation-only changes) are skipped rather
than forced into a bucket.
"""

from typing import Optional

# Must stay in sync with RefactoringGenerator.refactoring_types
REFACTORING_TYPE_TO_ID = {
    "extract_method": 0,
    "move_class": 1,
    "rename_variable": 2,
    "consolidate_duplicate_code": 3,
    "remove_dead_code": 4,
    "simplify_condition": 5,
    "split_class": 6,
    "extract_interface": 7,
    "reduce_coupling": 8,
    "improve_naming": 9,
}

# Ordered (first match wins) substrings of RefactoringMiner's "type" field,
# lowercased, mapped to our category names.
_RM_SUBSTRING_RULES = [
    ("extract interface", "extract_interface"),
    ("extract superclass", "split_class"),
    ("extract subclass", "split_class"),
    ("extract class", "split_class"),
    ("extract and move method", "extract_method"),
    ("extract method", "extract_method"),
    ("inline method", "consolidate_duplicate_code"),
    ("merge method", "consolidate_duplicate_code"),
    ("rename local variable", "rename_variable"),
    ("rename parameter", "rename_variable"),
    ("rename attribute", "rename_variable"),
    ("rename variable", "rename_variable"),
    ("move and rename class", "move_class"),
    ("rename class", "move_class"),
    ("move class", "move_class"),
    ("remove method", "remove_dead_code"),
    ("remove attribute", "remove_dead_code"),
    ("remove class", "remove_dead_code"),
    ("remove parameter", "remove_dead_code"),
    ("move and rename method", "reduce_coupling"),
    ("move method", "reduce_coupling"),
    ("move attribute", "reduce_coupling"),
    ("pull up method", "reduce_coupling"),
    ("push down method", "reduce_coupling"),
    ("rename method", "improve_naming"),
    ("consolidate conditional", "simplify_condition"),
    ("decompose conditional", "simplify_condition"),
    ("replace conditional", "simplify_condition"),
    ("invert condition", "simplify_condition"),
]


def map_rm_type(rm_type: str) -> Optional[str]:
    """Map a RefactoringMiner type string to one of our 10 category names, or None if unmapped"""
    if not rm_type:
        return None
    lowered = rm_type.lower()
    for substring, category in _RM_SUBSTRING_RULES:
        if substring in lowered:
            return category
    return None


def map_rm_type_to_id(rm_type: str) -> Optional[int]:
    """Map a RefactoringMiner type string directly to our numeric class ID"""
    category = map_rm_type(rm_type)
    return REFACTORING_TYPE_TO_ID.get(category) if category else None
