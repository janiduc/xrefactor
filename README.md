# XRefactor: Explainable AI-driven Cross-file Code Refactoring Framework

## Overview

XRefactor is an intelligent IDE framework designed to automate semantic, cross-file code refactoring by combining deep structural reasoning with transparent, human-readable explanations.

### Four Core Stages

1. **Code Property Graph (CPG) Construction** - Unifies syntax, control flow, and inter-file dependencies
2. **Graph Neural Network (GNN) Reasoning** - Models interactions of multiple entities using dependency hypergraphs
3. **Transformer Code Generation** - Generates refactored code conditioned on structural understanding
4. **Explainable AI (XAI) Module** - Provides decision rationales through dual-view causal inference

## Project Structure

```
xrefactor/
├── src/
│   ├── cpg/                    # Stage 1: CPG Construction
│   │   ├── cpg_builder.py      # Main CPG construction logic
│   │   └── __init__.py
│   ├── gnn/                    # Stage 2: GNN Reasoning
│   │   ├── gnn_model.py        # GNN architectures (GAT, GCN, GraphSAGE)
│   │   └── __init__.py
│   ├── transformer/            # Stage 3: Code Generation
│   │   ├── code_generator.py   # Transformer-based code generator
│   │   └── __init__.py
│   ├── xai/                    # Stage 4: Explainability
│   │   ├── explanation_module.py # XAI and evidence generation
│   │   └── __init__.py
│   ├── utils/                  # Utility modules
│   │   ├── helpers.py          # Configuration, logging, metrics
│   │   └── __init__.py
│   ├── core/                   # Core pipeline
│   │   ├── pipeline.py         # Pipeline orchestrator
│   │   └── __init__.py
│   └── __init__.py
├── configs/
│   └── config.yaml             # Configuration file
├── datasets/
│   └── preprocessed/           # Preprocessed datasets
├── models/                     # Trained models
├── experiments/                # Experiment tracking
├── tests/                      # Unit tests
├── outputs/                    # Pipeline outputs
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support, optional)

### Setup

```bash
# Clone the repository
cd xrefactor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

## Usage

### Quick Start

```bash
python main.py \
    --data-dir ../Data \
    --config ./configs/config.yaml \
    --output ./outputs \
    --device cuda
```

### Detailed Parameters

```bash
python main.py --help

Options:
  --data-dir DIR           Root directory containing source code (required)
  --config FILE            Path to config file (default: configs/config.yaml)
  --output DIR             Output directory (default: ./outputs)
  --device {cuda,cpu}      Device to use (default: cuda)
  --log-level {DEBUG,INFO,WARNING,ERROR}  Logging level (default: INFO)
```

### Python API

```python
from src.core.pipeline import XRefactorPipeline

# Initialize pipeline
pipeline = XRefactorPipeline(
    config_path="./configs/config.yaml",
    device="cuda"
)

# Run complete pipeline
results = pipeline.run_pipeline(
    data_directory="../Data",
    output_directory="./outputs"
)

# Access results
print(f"Status: {results['status']}")
print(f"Metrics: {results['metrics']}")
```

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
cpg:
  language: "java"              # Source language
  include_data_flow: true       # Include data flow analysis
  include_control_flow: true    # Include control flow analysis
  include_call_graph: true      # Include call graph

gnn:
  model_type: "gat"             # Model: gat, gcn, graphsage
  hidden_dims: [256, 256]       # Hidden layer dimensions
  output_dim: 128               # Output dimension
  num_heads: 8                  # Attention heads
  dropout: 0.1                  # Dropout rate

transformer:
  model_type: "codebert"        # Pre-trained model
  max_seq_length: 512           # Maximum sequence length
  hidden_size: 768              # Hidden size
  num_layers: 6                 # Number of layers

xai:
  method: "causal_inference"    # Explanation method
  explanation_length: 200       # Explanation length (words)
  num_evidence_cards: 3         # Number of evidence cards
  confidence_threshold: 0.7     # Confidence threshold
```

## Supported Languages

- Java (primary)
- Python
- C (planned)
- JavaScript (planned)

## Supported Refactoring Patterns

1. Extract Method
2. Move Class
3. Rename Variable
4. Consolidate Duplicate Code
5. Remove Dead Code
6. Simplify Condition
7. Split Class
8. Extract Interface
9. Reduce Coupling
10. Improve Naming

## Output

The pipeline generates:

- `pipeline_results_<timestamp>.json` - Complete pipeline results
- `metrics_<timestamp>.json` - Performance metrics
- `xrefactor.log` - Execution log
- Evidence cards with explanations
- Refactoring recommendations with confidence scores

## Example Output

```
Evidence Card #1
- Problem: Code Duplication
- Affected Files: FileA.java, FileB.java
- Recommendation: Extract method consolidateValidation()
- Confidence: 85%
- Explanation: Code duplication detected in validation logic...
```

## Performance Metrics

The pipeline tracks:

- CPG construction time and statistics
- GNN embedding quality
- Transformer generation accuracy
- XAI explanation coverage

See `outputs/metrics_<timestamp>.json` for detailed metrics.

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## Research Paper References

- Graph Neural Networks: [Kipf & Welling, 2016]
- Attention Mechanisms: [Vaswani et al., 2017]
- Code Understanding: [Allamanis et al., 2018]
- Explainability: [Ribeiro et al., 2016]

## License

This project is part of an MSc research initiative.

## Contact

For questions or issues, please refer to the research team documentation.

## Dataset

Sample datasets included:
- `Data/1273091433/jeesite/` - Java microservice project
- `Data/1725762388/axbushu/` - Multi-module Java project
- `Data/4pxzhou/FEBS-Cloud/` - Cloud-native Java project

## Future Enhancements

- [ ] Support for more programming languages
- [ ] Interactive web UI for explanations
- [ ] Real-time IDE plugin integration
- [ ] Machine learning model optimization
- [ ] Distributed processing for large codebases
- [ ] Integration with version control systems
