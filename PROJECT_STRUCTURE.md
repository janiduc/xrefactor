# XRefactor Project Structure

## Complete Directory Layout

```
xrefactor/
│
├── src/                                # Main source code
│   ├── __init__.py
│   │
│   ├── cpg/                            # Stage 1: Code Property Graph
│   │   ├── __init__.py
│   │   └── cpg_builder.py              # CPG construction
│   │       - CodeNode: Represents code entities
│   │       - CodeEdge: Represents relationships
│   │       - CodePropertyGraph: Main builder
│   │       - Support for Java, Python
│   │       - AST, CFG, DFG, Call graph extraction
│   │
│   ├── gnn/                            # Stage 2: Graph Neural Network
│   │   ├── __init__.py
│   │   └── gnn_model.py                # GNN implementations
│   │       - HypergraphGNN: Main GNN model (GAT/GCN/GraphSAGE)
│   │       - DependencyHypergraphEncoder: Node feature encoding
│   │       - RefactoringPredictor: Prediction head
│   │       - GNNTrainer: Training utilities
│   │       - convert_cpg_to_geometric_data: Format conversion
│   │
│   ├── transformer/                    # Stage 3: Code Generation
│   │   ├── __init__.py
│   │   └── code_generator.py           # Transformer models
│   │       - CodeTransformer: Main generator (CodeBERT/GraphCodeBERT)
│   │       - TransformerDecoder: Decode module
│   │       - RefactoringGenerator: High-level interface
│   │       - Support for greedy/beam search generation
│   │
│   ├── xai/                            # Stage 4: Explainability
│   │   ├── __init__.py
│   │   └── explanation_module.py       # XAI implementations
│   │       - EvidenceCard: Structured evidence
│   │       - CausalInferenceModule: Dual-view analysis
│   │       - ExplanationGenerator: Natural language explanations
│   │       - ExplainabilityReport: Comprehensive reports
│   │
│   ├── utils/                          # Utility modules
│   │   ├── __init__.py
│   │   └── helpers.py                  # Helper utilities
│   │       - ConfigManager: YAML configuration loading
│   │       - LoggerSetup: Logging configuration
│   │       - DatasetAnalyzer: Repository analysis
│   │       - MetricsTracker: Performance metrics
│   │
│   └── core/                           # Core pipeline
│       ├── __init__.py
│       └── pipeline.py                 # Pipeline orchestrator
│           - XRefactorPipeline: Main pipeline coordinator
│           - Manages all 4 stages
│           - Results aggregation and saving
│
├── configs/
│   └── config.yaml                     # Configuration file
│       - CPG settings (language, analysis types)
│       - GNN settings (model type, dims, layers)
│       - Transformer settings (model, seq length)
│       - Training settings (batch size, learning rate)
│       - XAI settings (method, explanation length)
│       - Data/Project paths
│
├── datasets/
│   └── preprocessed/                   # Preprocessed data
│       - Serialized CPGs
│       - Cached embeddings
│       - Training/validation splits
│
├── models/                             # Trained models
│   ├── gnn_models/                     # GNN checkpoints
│   ├── transformer_models/             # Transformer checkpoints
│   └── metadata.json                   # Model information
│
├── experiments/                        # Experiment tracking
│   ├── exp_001/                        # Experiment results
│   │   ├── config.yaml                 # Experiment config
│   │   ├── results.json                # Results
│   │   └── metrics.json                # Metrics
│   └── ...
│
├── tests/                              # Unit tests
│   ├── __init__.py
│   ├── test_core.py                    # Core module tests
│   │   - TestCPG: CPG tests
│   │   - TestGNN: GNN tests
│   │   - TestTransformer: Transformer tests
│   │   - TestXAI: XAI tests
│   └── conftest.py                     # Pytest configuration
│
├── outputs/                            # Pipeline outputs
│   ├── pipeline_results_TIMESTAMP.json # Results
│   ├── metrics_TIMESTAMP.json          # Metrics
│   ├── xrefactor.log                   # Execution log
│   └── evidence_cards_TIMESTAMP.json   # Evidence cards
│
├── main.py                             # Entry point script
├── quickstart.py                       # Quick start/validation
├── examples.py                         # Usage examples
│
├── requirements.txt                    # Python dependencies
├── README.md                           # Project overview
├── ARCHITECTURE.md                     # Architecture documentation
├── DEVELOPMENT.md                      # Development guide
├── PROJECT_STRUCTURE.md                # This file
│
└── .gitignore                          # Git ignore rules
```

## File Dependencies

```
main.py
  └─> src.core.pipeline.XRefactorPipeline
      ├─> src.cpg.cpg_builder.CodePropertyGraph
      ├─> src.gnn.gnn_model.HypergraphGNN
      ├─> src.transformer.code_generator.CodeTransformer
      ├─> src.xai.explanation_module.CausalInferenceModule
      └─> src.utils.helpers.ConfigManager, LoggerSetup

src.core.pipeline.XRefactorPipeline
  ├─> src.cpg.cpg_builder
  ├─> src.gnn.gnn_model
  ├─> src.transformer.code_generator
  ├─> src.xai.explanation_module
  └─> src.utils.helpers
```

## Key Classes and Their Responsibilities

### CPG Module (cpg/cpg_builder.py)

| Class | Responsibility |
|-------|-----------------|
| `CodeNode` | Represents individual code elements |
| `CodeEdge` | Represents relationships between nodes |
| `CodePropertyGraph` | Builds and maintains the property graph |

### GNN Module (gnn/gnn_model.py)

| Class | Responsibility |
|-------|-----------------|
| `HypergraphGNN` | Core GNN model implementation |
| `DependencyHypergraphEncoder` | Encodes nodes to features |
| `RefactoringPredictor` | Predicts refactoring type |
| `GNNTrainer` | Handles training loop |

### Transformer Module (transformer/code_generator.py)

| Class | Responsibility |
|-------|-----------------|
| `CodeTransformer` | Main code generation model |
| `TransformerDecoder` | Decoding component |
| `RefactoringGenerator` | High-level generation interface |

### XAI Module (xai/explanation_module.py)

| Class | Responsibility |
|-------|-----------------|
| `EvidenceCard` | Structured evidence representation |
| `CausalInferenceModule` | Dual-view causal analysis |
| `ExplanationGenerator` | Generates explanations |
| `ExplainabilityReport` | Creates comprehensive reports |

### Utils Module (utils/helpers.py)

| Class | Responsibility |
|-------|-----------------|
| `ConfigManager` | YAML config loading/management |
| `LoggerSetup` | Logging configuration |
| `DatasetAnalyzer` | Analyzes repositories |
| `MetricsTracker` | Tracks metrics |

### Core Module (core/pipeline.py)

| Class | Responsibility |
|-------|-----------------|
| `XRefactorPipeline` | Orchestrates all 4 stages |

## Data Flow

```
Input: Source Code Repository
        ↓
    Stage 1: CPG Construction
    ├─ Output: CodePropertyGraph object
    ├─ Saved as: datasets/preprocessed/cpg_*.pkl
    │
    Stage 2: GNN Reasoning
    ├─ Input: CPG graph
    ├─ Output: Node embeddings, graph embeddings, predictions
    ├─ Saved as: models/gnn_models/embeddings_*.pt
    │
    Stage 3: Transformer Generation
    ├─ Input: Embeddings, source code
    ├─ Output: Refactored code candidates
    ├─ Saved as: outputs/refactored_code_*.txt
    │
    Stage 4: XAI Explanation
    ├─ Input: Predictions, embeddings
    ├─ Output: Evidence cards, explanations
    ├─ Saved as: outputs/evidence_cards_*.json
    │
Output: Refactoring Suggestions + Explanations
```

## Configuration Files

### config.yaml Structure

```yaml
project:
  data_dir: "../Data"
  output_dir: "./outputs"
  model_dir: "./models"

cpg:
  language: "java"
  include_data_flow: true
  include_control_flow: true
  include_call_graph: true

gnn:
  model_type: "gat"
  hidden_dims: [256, 256]
  output_dim: 128
  num_layers: 3

transformer:
  model_type: "codebert"
  hidden_size: 768
  num_layers: 6

xai:
  method: "causal_inference"
  num_evidence_cards: 3
  confidence_threshold: 0.7

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100
  device: "cuda"
```

## Output Structure

```
outputs/
├── pipeline_results_20240514_143022.json
│   ├── timestamp
│   ├── status
│   ├── stages
│   │   ├── cpg
│   │   ├── gnn
│   │   ├── transformer
│   │   └── xai
│   └── metrics
│
├── metrics_20240514_143022.json
│   ├── cpg_*
│   ├── gnn_*
│   ├── transformer_*
│   └── xai_*
│
├── evidence_cards_20240514_143022.json
│   ├── [array of evidence cards]
│   └── [with explanations and metrics]
│
└── xrefactor.log
    ├── INFO logs
    ├── DEBUG logs (if enabled)
    └── ERROR logs
```

## Module Initialization Order

1. `ConfigManager` - Load configuration
2. `LoggerSetup` - Setup logging
3. `CodePropertyGraph` - Initialize CPG
4. `HypergraphGNN` - Initialize GNN
5. `CodeTransformer` - Initialize transformer
6. `CausalInferenceModule` - Initialize XAI
7. `XRefactorPipeline` - Create pipeline
8. Run stages sequentially

## Performance Characteristics

| Component | Memory | Time | Scalability |
|-----------|--------|------|-------------|
| CPG Construction | O(n) | O(n log n) | Linear |
| GNN Forward Pass | O(n) | O(n) | GPU-limited |
| Transformer Generation | O(seq_len²) | O(seq_len) | Quadratic |
| XAI Analysis | O(batch_size) | O(batch_size) | Linear |

where n = number of nodes in graph, seq_len = sequence length

## Extension Guidelines

### Adding a New Language Parser

1. Create new parser in `cpg/cpg_builder.py`: `_process_{language}_file()`
2. Define language-specific node/edge types
3. Update `_process_file()` to dispatch to new parser
4. Add language option to config.yaml

### Adding a New GNN Architecture

1. Implement layer in `gnn/gnn_model.py`
2. Update `HypergraphGNN` to support new type
3. Add to config options
4. Create tests

### Adding New Refactoring Pattern

1. Increase `num_refactoring_types` in predictor
2. Add pattern name to refactoring_types dict
3. Add explanation templates
4. Add training data examples

## Testing

```bash
pytest tests/              # Run all tests
pytest tests/ -v          # Verbose output
pytest tests/ --cov       # With coverage
```

## Version Control

- Use semantic versioning: MAJOR.MINOR.PATCH
- Tag releases: `v0.1.0`
- Branch naming: `feature/description`, `bugfix/description`
