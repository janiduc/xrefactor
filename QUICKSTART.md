# XRefactor Quick Reference

## Installation & Setup (5 minutes)

```bash
# 1. Navigate to project
cd xrefactor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python quickstart.py
```

## File Structure at a Glance

```
xrefactor/
├── src/                    ← All source code
│   ├── cpg/               ← Stage 1: Code property graphs
│   ├── gnn/               ← Stage 2: Neural networks
│   ├── transformer/       ← Stage 3: Code generation
│   ├── xai/               ← Stage 4: Explanations
│   ├── utils/             ← Helpers & utilities
│   └── core/              ← Pipeline orchestrator
├── configs/
│   └── config.yaml        ← Configuration (modify here!)
├── tests/
│   └── test_core.py       ← Unit tests
├── outputs/               ← Results saved here
├── main.py                ← Entry point
└── README.md              ← Full documentation
```

## Running XRefactor

### Quick Test
```bash
python quickstart.py --full
```

### Full Pipeline
```bash
python main.py \
    --data-dir ../Data \
    --output ./outputs \
    --device cuda
```

### Custom Configuration
1. Edit `configs/config.yaml`
2. Run: `python main.py --config ./configs/config.yaml --data-dir ../Data`

## Core Modules

### 1. CPG Construction
```python
from src.cpg.cpg_builder import CodePropertyGraph

cpg = CodePropertyGraph(language="java")
cpg.build_from_directory("../Data/1273091433/jeesite")
stats = cpg.get_statistics()
print(f"Nodes: {stats['total_nodes']}")
```

### 2. GNN Model
```python
from src.gnn.gnn_model import HypergraphGNN

model = HypergraphGNN(
    input_dim=1,
    hidden_dims=[256, 256],
    output_dim=128,
    model_type="gat"  # or "gcn", "graphsage"
)

node_embeddings, graph_embedding = model(x, edge_index)
```

### 3. Code Generation
```python
from src.transformer.code_generator import RefactoringGenerator

generator = RefactoringGenerator(transformer_model)
suggestion = generator.suggest_refactoring(
    source_code=code,
    gnn_embeddings=embedding,
    confidence_score=0.85
)
```

### 4. XAI Explanations
```python
from src.xai.explanation_module import ExplainabilityReport

report = ExplainabilityReport(explanation_generator)
result = report.generate_report(evidence_cards)
print(report.format_report(result))
```

## Configuration Reference

### CPG Settings
```yaml
cpg:
  language: "java"                    # java, python, etc.
  include_data_flow: true             # Enable data flow
  include_control_flow: true          # Enable control flow
  include_call_graph: true            # Enable call graph
```

### GNN Settings
```yaml
gnn:
  model_type: "gat"                   # gat, gcn, graphsage
  hidden_dims: [256, 256]             # Hidden layer sizes
  output_dim: 128                     # Output dimension
  num_heads: 8                        # For GAT
  num_layers: 3                       # Number of layers
  dropout: 0.1                        # Dropout rate
```

### Transformer Settings
```yaml
transformer:
  model_type: "codebert"              # Pre-trained model
  max_seq_length: 512                 # Max token length
  hidden_size: 768                    # Hidden dimension
  num_layers: 6                       # Decoder layers
```

### XAI Settings
```yaml
xai:
  method: "causal_inference"          # Explanation method
  explanation_length: 200             # Max word count
  num_evidence_cards: 3               # Cards per suggestion
  confidence_threshold: 0.7           # Min confidence
```

## Common Tasks

### Run on Specific Repository
```bash
python main.py --data-dir ../Data/1273091433/jeesite --output ./outputs
```

### Use CPU Instead of GPU
```bash
python main.py --data-dir ../Data --device cpu
```

### Enable Debug Logging
```bash
python main.py --data-dir ../Data --log-level DEBUG
```

### Save Custom Config
```python
from src.utils.helpers import ConfigManager

config = ConfigManager("./configs/config.yaml")
config.set("gnn.hidden_dims", [512, 512])
config.save("./configs/custom_config.yaml")
```

## Output Files

After running pipeline:

```
outputs/
├── pipeline_results_20240514_143022.json
│   └── All pipeline results and metrics
├── metrics_20240514_143022.json
│   └── Performance metrics
├── evidence_cards_20240514_143022.json
│   └── Refactoring suggestions + explanations
└── xrefactor.log
    └── Execution log
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run: `pip install -r requirements.txt` |
| `CUDA out of memory` | Reduce batch size in config or use `--device cpu` |
| `Data not found` | Verify `../Data` directory exists with samples |
| `Config file not found` | Use `--config` with full path |
| Tests failing | Run `pip install pytest` and retry |

## Supported Languages

- ✓ Java (primary)
- ✓ Python (partial)
- ⏳ C/C++ (planned)
- ⏳ JavaScript (planned)

## Refactoring Patterns

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

## Key Classes

| Module | Class | Purpose |
|--------|-------|---------|
| CPG | `CodePropertyGraph` | Build code graphs |
| GNN | `HypergraphGNN` | Neural network reasoning |
| Transformer | `CodeTransformer` | Generate code |
| XAI | `CausalInferenceModule` | Explain decisions |
| Core | `XRefactorPipeline` | Orchestrate pipeline |
| Utils | `ConfigManager` | Manage config |

## Documentation Files

- `README.md` - Full overview and usage
- `ARCHITECTURE.md` - System design
- `DEVELOPMENT.md` - Dev environment setup
- `PROJECT_STRUCTURE.md` - Directory layout
- `IMPLEMENTATION_SUMMARY.md` - What was built

## Testing

```bash
# Run all tests
pytest tests/

# Specific test
pytest tests/test_core.py::TestCPG -v

# With coverage
pytest tests/ --cov=src
```

## Performance Tips

1. Use GPU: `--device cuda`
2. Reduce batch size for large codebases
3. Cache CPG results between runs
4. Use parallel processing for file parsing

## Next Steps

1. ✓ Installation complete
2. ✓ Run `python quickstart.py --full`
3. ✓ Review `configs/config.yaml`
4. ✓ Run `python main.py --data-dir ../Data --output ./outputs`
5. ✓ Check `outputs/` for results
6. → Train models on your datasets
7. → Evaluate on benchmark tasks
8. → Integrate with IDE plugins

## Getting Help

1. Check `README.md` for detailed documentation
2. Review `DEVELOPMENT.md` for development guide
3. Look at `examples.py` for usage patterns
4. Check logs in `outputs/xrefactor.log`
5. Run tests to verify installation

## Quick Links

- **Main Entry:** `main.py`
- **Configuration:** `configs/config.yaml`
- **Pipeline:** `src/core/pipeline.py`
- **Tests:** `tests/test_core.py`
- **Examples:** `examples.py`
- **Docs:** README.md, ARCHITECTURE.md, DEVELOPMENT.md

---

**Status:** Ready to use! 🚀

Start with: `python quickstart.py --full`
