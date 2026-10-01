"""
Deterministic, AST-driven Java refactoring.

Where Stage 3's transformer *generates* code (and may be wrong), this package
*transforms* code by splicing the original source at positions derived from
javalang's AST and token stream. javalang has no unparser, so every refactoring
here is a text edit on the original bytes - which also means formatting and
comments survive untouched.

Modules:
    offsets  - (line, column) <-> character offset, cached tokenization
    extent   - source extents of members and statements, via token brace matching
"""
