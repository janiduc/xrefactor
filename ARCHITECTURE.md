"""
Architecture and design documentation for XRefactor
"""

# XRefactor Architecture

## Current Status

The architecture below was the original design and remains accurate as *design* - what has changed since is that every stage now runs on real trained weights by default, not scaffolding. See `README.md`'s "Current Status" table for headline numbers and `DEVELOPMENT.md`'s "Known Limitations" for the honest caveats (dataset sizes, thin classes, the transformer's undertrained repetition, the XAI module's weak-supervision framing). One structural addition beyond the original design: `src/cpg/smell_detector.py` adds a rule-based code-smell detection layer (Stage 1) that runs alongside the GNN's learned `refactoring_type` prediction (Stage 2) - Stage 3 cross-checks the two and surfaces disagreement rather than silently picking one, closing the loop from "a specific code smell was identified" to "code was generated for that exact pattern."

One correction to the original design, rather than an addition: the CPG's call graph was never actually inter-file. Every call edge was synthesised against the *caller's* file path, so cross-file invocations pointed at node ids that did not exist, and the "inter-file dependency resolution" the diagram below claims was not happening. `src/cpg/call_resolver.py` now resolves invocations against a global index of every declaration in the corpus, in a second pass - see the CPG section's "Call-graph resolution" below, and `DEVELOPMENT.md`'s "Known Limitations" for measured coverage and what remains out of reach without a classpath.

## High-Level Overview

```
Input Source Code (Multiple Files)
        ↓
    ┌───────────────────────────────────────┐
    │  Stage 1: CPG Construction            │
    │  - Parse source code                  │
    │  - Extract AST, CFG, DFG              │
    │  - Build dependency graph             │
    └──────────────┬────────────────────────┘
                   ↓
            Code Property Graph
                   ↓
    ┌───────────────────────────────────────┐
    │  Stage 2: GNN Reasoning               │
    │  - Encode nodes                       │
    │  - Apply GNN (GAT/GCN/GraphSAGE)      │
    │  - Generate embeddings                │
    │  - Predict refactoring type           │
    └──────────────┬────────────────────────┘
                   ↓
            Node & Graph Embeddings
                   ↓
    ┌───────────────────────────────────────┐
    │  Stage 3: Transformer Generation      │
    │  - Encode source code                 │
    │  - Fuse GNN context via attention     │
    │  - Generate refactored code           │
    └──────────────┬────────────────────────┘
                   ↓
            Refactored Code Candidates
                   ↓
    ┌───────────────────────────────────────┐
    │  Stage 4: XAI Explanation             │
    │  - Causal inference                   │
    │  - Generate evidence cards            │
    │  - Create explanations                │
    └──────────────┬────────────────────────┘
                   ↓
        Refactoring Suggestions with
        Explanations & Confidence Scores
```

## Component Architecture

### 1. Code Property Graph (CPG) Module

**Purpose:** Construct unified representation of codebase

**Components:**
- `CodeNode`: Represents code entities (classes, methods, fields)
- `CodeEdge`: Represents relationships (calls, uses, inherits)
- `CodePropertyGraph`: Main graph builder

**Edge types actually produced** (`CodeEdge.edge_type`, as emitted - this list
was previously aspirational and named three types the builder never created):
- `control_flow`: statement sequencing and branching. `metadata["kind"]` is one
  of `entry`, `sequence`, `branch_true`, `branch_false`, `loop_body`,
  `try_body`, `catch_body`, `finally_body`, `case_body`.
- `data_flow`: def-use. `metadata["role"]` is `parameter`, `definition` or `use`
  (i.e. "defines" and "uses" are roles on this edge type, not separate types).
- `calls`: method invocation, resolved to a real declaration by
  `src/cpg/call_resolver.py`. `metadata` carries `resolution` (which rule found
  the target), `receiver_kind` and `cross_file`.
- `inherits`: class-to-class `extends`/`implements`.

Containment is represented by `CodeNode.file_path` + `CodePropertyGraph.file_map`
rather than by an edge.

**Call-graph resolution (`src/cpg/call_resolver.py`).** Invocations are collected
during a first pass and resolved during a second, once every declaration in the
corpus is indexed. Each edge records the rule that produced it (`same_file`,
`inherited`, `super_chain`, `unique_global`, `static_class`, `qualifier_type`),
and every invocation that is NOT resolved is recorded with a specific reason
rather than dropped or guessed. Full type resolution needs the dependency
classpath these repos are not built with, so the resolver refuses instead of
inventing a target - see that module's docstring for the strategy ladder and
each deliberate refusal.

**Language Support:**
- Java: Full support via javalang parser
- Python: scaffold only - `build_from_directory` globs `*.{language}`, so
  `language="python"` looks for `*.python` and matches nothing. Class/function
  nodes are extracted but no edges are built.
- C/C++: Planned
- JavaScript: Planned

### 2. Graph Neural Network (GNN) Module

**Purpose:** Reason over dependency structure and predict refactoring needs

**Architecture Options:**
- **GAT (Graph Attention Networks)**: Multi-head attention for flexible relationship modeling
- **GCN (Graph Convolutional Networks)**: Simple convolution over neighbors
- **GraphSAGE**: Scalable inductive learning

**Pipeline:**
1. Encode node types into embeddings
2. Apply GNN layers (default 3 layers)
3. Global pooling for graph-level representation
4. Predict refactoring type and confidence

**Embeddings:**
- Node embeddings: [num_nodes, output_dim]
- Graph embeddings: [batch_size, output_dim]

### 3. Transformer Code Generation Module

**Purpose:** Generate semantically consistent refactored code

**Architecture:**
- **Encoder**: Pre-trained CodeBERT/GraphCodeBERT
- **Fusion Layer**: Multi-head attention to combine encoder and GNN outputs
- **Decoder**: Transformer decoder with position embeddings

**Generation Methods:**
- Greedy decoding (default)
- Beam search (planned)
- Temperature sampling (planned)

### 4. Explainable AI (XAI) Module

**Purpose:** Provide transparent reasoning for refactoring recommendations

**Components:**
- **Causal Inference Module**: Dual-view analysis
  - View 1: Problem detection in original code
  - View 2: Solution quality evaluation
- **Explanation Generator**: Natural language explanations
- **Evidence Cards**: Structured evidence linking changes to problems

**Evidence Card Contents:**
- Problem type (code smell, architectural violation, etc.)
- Affected entities (files, classes, methods)
- Natural language explanation
- Supporting metrics (problem score, solution score, improvement delta)
- Confidence score
- Specific recommendation

## Data Flow

```
Source Code Files
    ↓
CPG Construction
    ├─ AST Extraction
    ├─ CFG Analysis
    ├─ Call Graph Building
    └─ Dependency Resolution
    ↓
Graph Representation [NetworkX MultiDiGraph]
    ├─ Nodes: {node_id: CodeNode}
    ├─ Edges: [CodeEdge]
    └─ File Mapping: {file_path: [node_ids]}
    ↓
Geometric Conversion
    ├─ Node Features: [num_nodes, feature_dim]
    └─ Edge Index: [2, num_edges]
    ↓
GNN Processing
    ├─ Node Embeddings: [num_nodes, output_dim]
    ├─ Graph Embedding: [batch_size, output_dim]
    ├─ Problem Scores: [0, 1]
    └─ Confidence Scores: [0, 1]
    ↓
Transformer Generation
    ├─ Source Code Encoding
    ├─ Attention Fusion
    └─ Token Generation
    ↓
XAI Analysis
    ├─ Problem Detection
    ├─ Solution Evaluation
    ├─ Attention Visualization
    └─ Evidence Generation
    ↓
Output: Refactoring Suggestions + Explanations
```

## Key Design Patterns

1. **Pipeline Pattern**: Sequential stages with clear interfaces
2. **Factory Pattern**: Model creation (GNN types, transformers)
3. **Strategy Pattern**: Different explanation methods
4. **Observer Pattern**: Metrics tracking

## Extension Points

### Adding New Languages

1. Implement parser in CPG module
2. Add to language detection in `CodePropertyGraph._process_file()`
3. Define node and edge types for the language

### Adding New Refactoring Patterns

1. Add to `RefactoringPredictor` output classes
2. Create template in `ExplanationGenerator.solution_templates`
3. Update transformer training data

### Adding New Explanation Methods

1. Implement interface in `explanation_module.py`
2. Register in `ExplanationGenerator`
3. Add to configuration options

## Performance Considerations

1. **Memory**: Large codebases may require distributed processing
2. **Computation**: GNN inference is GPU-accelerated
3. **Latency**: Two-stage pipeline (offline training, online inference)
4. **Scalability**: Graph pooling reduces large graphs to manageable size

## Evaluation Metrics

- **Precision**: Correctness of generated refactorings
- **Recall**: Coverage of detectable issues
- **Explainability**: Quality and comprehensiveness of explanations
- **Inference Time**: Time to generate suggestions
- **User Trust**: Confidence in AI-generated suggestions
