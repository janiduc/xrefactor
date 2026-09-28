# XRefactor Development Guide

## Setup Development Environment

```bash
# Clone repository
cd xrefactor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 isort

# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Code Organization

### Module Structure

Each stage module has:
- Core implementation file
- `__init__.py` with public API
- Tests in `tests/` directory
- Examples and documentation

### Naming Conventions

- Classes: PascalCase (e.g., `CodePropertyGraph`)
- Functions/Methods: snake_case (e.g., `build_from_directory`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_FILE_SIZE`)
- Private attributes: _leading_underscore

## Development Workflow

### 1. Adding New Features

```python
# src/cpg/cpg_builder.py - Example

def new_feature(self, param: str) -> Dict[str, Any]:
    """
    Short description.
    
    Long description explaining the feature.
    
    Args:
        param: Description of parameter
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When X condition
    """
    logger.info(f"Starting new feature with {param}")
    
    result = {}
    
    logger.debug(f"Intermediate result: {result}")
    
    return result
```

### 2. Error Handling

```python
try:
    cpg.build_from_directory(data_dir)
except FileNotFoundError:
    logger.error(f"Directory not found: {data_dir}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

### 3. Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_core.py::TestCPG::test_cpg_initialization

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific level tests
pytest tests/ -k "cpg"  # Tests with 'cpg' in name
```

### 4. Code Quality

```bash
# Format code
black src/

# Check code style
flake8 src/

# Sort imports
isort src/
```

## Adding a New Refactoring Pattern

### Step 1: Update CPG Analysis

```python
# In cpg_builder.py
def _analyze_pattern(self, code: str) -> Dict[str, Any]:
    """Analyze for new pattern"""
    pass
```

### Step 2: Update GNN Output

```python
# In gnn_model.py
class RefactoringPredictor:
    def __init__(self, ..., num_refactoring_types: int = 11):  # Increase from 10
        pass
```

### Step 3: Update Transformer

```python
# In code_generator.py
self.refactoring_types = {
    ...
    10: "new_refactoring_pattern"
}
```

### Step 4: Add Explanation Template

```python
# In explanation_module.py
self.problem_templates["new_pattern"] = "..."
self.solution_templates["new_refactoring_pattern"] = "..."
```

### Step 5: Write Tests

```python
# In tests/test_core.py
def test_new_pattern():
    # Test implementation
    pass
```

## Debugging

### Enable Debug Logging

```python
from src.utils.helpers import LoggerSetup

LoggerSetup.setup(log_level="DEBUG")
```

### Using Debugger

```python
import pdb; pdb.set_trace()  # Drop debugger here
```

### Profiling Performance

```python
import cProfile

cProfile.run('pipeline.run_pipeline(...)')
```

## Common Issues and Solutions

### Issue: Out of Memory

**Solution**: Reduce batch size or process files in batches

```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: Slow Graph Construction

**Solution**: Enable parallel processing

```python
cpg.build_from_directory(
    directory_path,
    parallel=True,
    num_workers=4
)
```

### Issue: Model Not Converging

**Solution**: Check learning rate and gradient flow

```yaml
training:
  learning_rate: 0.0001  # Try different values
  warmup_steps: 1000     # Gradual warmup
```

## Contributing

### Before Submitting Code

1. Run tests: `pytest tests/`
2. Format code: `black src/`
3. Check style: `flake8 src/`
4. Update docstrings
5. Add test cases for new features
6. Update README if needed

### Commit Message Format

```
[Stage] Brief description

Longer explanation if needed.

Related to: #issue_number
```

Example:
```
[GNN] Add GraphSAGE model support

Implemented GraphSAGEConv layer for scalable neighbor sampling.
Added tests for GraphSAGE model initialization and forward pass.

Related to: #42
```

## Performance Optimization Tips

1. **Use GPU**: Set device to "cuda" for 10-100x speedup
2. **Batch Processing**: Process multiple files simultaneously
3. **Caching**: Cache parsed graphs during development
4. **Profiling**: Use cProfile to identify bottlenecks
5. **Quantization**: Reduce model precision for inference

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def method(param1: str, param2: int) -> Dict[str, Any]:
    """Short description.
    
    Longer description explaining what the method does,
    how it works, and any important details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
    
    Returns:
        Dictionary with keys:
            - key1: Description
            - key2: Description
    
    Raises:
        ValueError: When parameter validation fails
        RuntimeError: When operation fails
    
    Examples:
        >>> result = method("input", 42)
        >>> print(result)
        {'key1': 'value1'}
    """
```

### Module-Level Documentation

```python
"""
Module description - clear, concise overview.

This module handles [specific responsibility].

Key Classes:
    - ClassName: Description

Key Functions:
    - function_name: Description

Usage:
    from module import ClassName
    
    obj = ClassName()
    result = obj.method()
"""
```

## Known Limitations

Honest caveats behind the numbers in `README.md`'s status table, so they aren't mistaken for converged results.

**Mining/labeling (`refactoring_mining/`)**: 3,731 labeled examples across 10/10 classes from 4 repos (`mining_report.json` has the full per-repo breakdown), but `extract_interface` (8 examples) and `simplify_condition` (27 examples) are too thin to trust. RefactoringMiner's ~90 fine-grained types are deliberately mapped down to this project's 10-class taxonomy (`src/gnn/refactoring_types.py`); types with no mapping are dropped by design, not a bug.

**Refactoring predictor (Stage 2 classifier)**: 0.51 test accuracy / 0.25 macro F1 on the held-out split. Classes with reasonable support show real F1 (`move_class` 0.70, `rename_variable` 0.65, `reduce_coupling` 0.51, `remove_dead_code` 0.43, `extract_method` 0.19); the thinnest classes are at 0 F1 - genuinely too little data, not a training bug (see `models/refactoring_predictor_trained.metrics.json`).

**Code-smell detector (Stage 1, `src/cpg/smell_detector.py`)**: all 6 detectors are deliberately simple heuristics, not static-analysis-grade. Spot-checked against real history: the `long_method` detector shows a real ~2x lift (13.3% hit rate on actual Extract Method targets vs. 6.7% population base rate) on a 15-example sample - directionally real, not a rigorous benchmark. The `dead_code` detector has a known blind spot: it only sees same-repo call edges the CPG builder itself tracks, so framework-invoked methods (e.g. Spring `@RequestMapping` handlers, `@Bean` factories) routinely look "dead" when they aren't - confirmed on a held-out repo where `dead_code` was ~90% of all findings.

**Transformer (Stage 3)**: trained on only 1,000 of 2,612 available before/after pairs, 2 epochs, bounded by CPU wall-clock time in the session that trained it. Conditioned on a learned per-*pattern* placeholder vector standing in for real per-example GNN context (see `src/transformer/train_transformer.py`'s docstring) - a documented, drop-in-replaceable simplification. Output moved from incoherent subword garbage (a real vocab-size/special-token bug, now fixed) to real-but-often-repetitive text (a separate, known undertrained-greedy-decoding failure mode). Structural pass rate (`src/transformer/pattern_validators.py`, `outputs/evaluation_report_*.json`) is 13.3% overall on held-out pairs - the honest number after a shared degeneracy guard was added specifically because two validators were initially rubber-stamping repetitive garbage as "passing."

**XAI (Stage 4, `src/xai/train_causal_module.py`)**: template-based explanation *text* is a deliberate, retained design choice - the templates are reasonably grounded once Stage 2/3 predictions are real. The two learned scoring heads are weak/proxy-supervised (not ground truth): `problem_detector` reached 0.88-0.95 accuracy (real signal - refactoring targets are structurally distinctive), but `solution_evaluator` stayed near chance (~0.4-0.6). Likely cause: the GNN embeddings feeding both heads are purely structural/type-based with no semantic text content, so many before/after pairs (especially local changes like renames) are close to indistinguishable in this embedding space - a real, unresolved limitation, not something papered over. No automatic metric substitutes for a human rubric on evidence-card quality; `evaluate.py` recommends one rather than fabricating a number.

**Known cross-stage disagreement**: on a held-out repo never used in mining/training, the rule-based smell detector and the GNN's learned prediction disagreed on 5/5 traced suggestions (smell said `remove_dead_code`, GNN said `reduce_coupling`, consistently) - see `outputs/evaluation_report_*.json`'s `end_to_end_trace`. This is presented as a real, reproducible finding about where the two views currently diverge, not resolved in either direction.

## Resources

- [PyTorch Documentation](https://pytorch.org/docs)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Transformers Documentation](https://huggingface.co/transformers/)
- [NetworkX Guide](https://networkx.org/documentation/)

## Getting Help

1. Check existing issues and documentation
2. Review similar code in codebase
3. Consult ARCHITECTURE.md for design patterns
4. Create detailed issue with minimal reproducible example
