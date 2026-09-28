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
