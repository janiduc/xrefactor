# XRefactor Testing Guide

Comprehensive testing strategy for XRefactor pipeline with GUI integration.

## Quick Start

```powershell
cd "d:\MSc. IT\Year2 Sem1\Research Project\mscResearchXrefactor\xrefactor"
.\.venv\Scripts\Activate.ps1

# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Test Organization

### Test Levels (from DEVELOPMENT.md)

**See [DEVELOPMENT.md § Testing](DEVELOPMENT.md#3-testing)** for the base testing command structure.

### Test Files

```
tests/
├── test_core.py             # Stage 1-4 pipeline tests (existing)
├── test_web.py               # Flask API endpoint tests
├── test_integration.py       # End-to-end integration tests
├── test_utils.py             # data_split / metrics utility tests
├── test_smell_detector.py    # one positive/negative case per code-smell detector
├── test_pattern_validators.py # one passing/failing case per refactoring-pattern validator
├── test_checkpoints.py       # all 4 checkpoint-loading paths, with real (tiny) checkpoints
└── fixtures/
    └── sample_code/     # Test data fixtures
```

`test_checkpoints.py` is worth calling out specifically: it's the only suite that loads an actually-non-null checkpoint (a tiny, synthetic one with deliberately distinctive constant weights) into each of the 4 `_load_*_checkpoint_if_available` methods and asserts the target module's weights changed. To regenerate its fixtures by hand (e.g. while debugging a loader), the pattern is:
```python
distinctive_module = SameArchitectureClass(...)
with torch.no_grad():
    for p in distinctive_module.parameters():
        p.fill_(0.1234)  # any distinctive constant
torch.save({"some_state_key": distinctive_module.state_dict()}, tmp_path / "test.pt")
```
then load it via the pipeline's loader and assert the live module's weights equal that constant.

---

## Phase 4: Complete Test Suite

### 4a. Unit Tests (Stage Pipeline)

Existing tests for each stage (see ARCHITECTURE.md § Data Flow):

```powershell
# Test CPG construction (Stage 1)
pytest tests/test_core.py::TestCPG -v

# Test GNN model (Stage 2)
pytest tests/test_core.py::TestGNN -v

# Test Transformer (Stage 3)
pytest tests/test_core.py::TestTransformer -v

# Test XAI module (Stage 4)
pytest tests/test_core.py::TestXAI -v
```

**Expected Output:**
```
test_cpg_initialization PASSED
test_cpg_build_from_directory PASSED
test_gnn_forward_pass PASSED
...
```

### 4b. Web/API Tests (NEW)

Tests for Flask endpoints and REST API (see src/web/api.py):

```powershell
# All API tests
pytest tests/test_web.py -v

# Specific endpoint tests
pytest tests/test_web.py::TestAPIHealth -v
pytest tests/test_web.py::TestAPIEndpoints -v
pytest tests/test_web.py::TestErrorHandling -v
```

**Expected Output:**
```
test_health_check PASSED
test_get_pipeline_stages PASSED
test_get_supported_languages PASSED
...
```

### 4c. Integration Tests (NEW)

End-to-end tests combining components:

```powershell
# All integration tests
pytest tests/test_integration.py -v

# Specific test classes
pytest tests/test_integration.py::TestPipelineIntegration -v
pytest tests/test_integration.py::TestComponentInteraction -v
pytest tests/test_integration.py::TestErrorRecovery -v
```

**Expected Output:**
```
test_pipeline_initialization PASSED
test_gnn_model_creation PASSED
test_cpg_to_gnn_flow PASSED
...
```

---

## Phase 5: End-to-End Testing

### 5a. Start Backend Server

```powershell
# Terminal 1: Start Flask app
cd "d:\MSc. IT\Year2 Sem1\Research Project\mscResearchXrefactor\xrefactor"
.\.venv\Scripts\Activate.ps1
python app.py --host 127.0.0.1 --port 5000 --debug
```

**Expected Output:**
```
================================================================================
XREFACTOR - EXPLAINABLE AI-DRIVEN CROSS-FILE CODE REFACTORING FRAMEWORK
================================================================================
Web Application Starting...
Host: 127.0.0.1
Port: 5000
Debug Mode: True

Open your browser and navigate to:
http://127.0.0.1:5000
```

### 5b. Open GUI in Browser

```
Open: http://127.0.0.1:5000
```

**Expected:**
- Header with "XRefactor" title
- Sidebar navigation (Home, Pipeline, Configuration, Results, Documentation)
- Status badge showing "● Connected"
- 4-stage pipeline diagram

### 5c. Test Each View

#### **Home Tab:**
- ✓ See 4-stage pipeline diagram
- ✓ See quick start guide
- ✓ Status shows "● Connected"

#### **Pipeline Tab:**
1. Enter source code directory:
   ```
   ../Data/1273091433/jeesite
   ```
2. Select device: `CPU`
3. Click **"▶️ Run Full Pipeline"**
4. Wait for completion (1-5 minutes depending on code size)
5. View results in output log

**Expected:**
```json
{
  "status": "success",
  "message": "Pipeline executed successfully",
  "results": {
    "stages": {
      "stage_1": { "nodes": 3933, "edges": 9756 },
      "stage_2": { "embeddings_shape": [3933, 128] },
      "stage_3": { "refactored_code": [...] },
      "stage_4": { "explanations": [...] }
    }
  }
}
```

#### **Configuration Tab:**
1. Review default settings (Java, GAT, 256x256 hidden dims)
2. Change GNN model: GAT → GCN
3. Click **"💾 Save Configuration"**
4. Reload page: settings should persist
5. Click **"📂 Load Configuration"**

#### **Results Tab:**
1. Run a pipeline
2. Go to Results tab
3. View formatted results with statistics

#### **Documentation Tab:**
- ✓ See architecture overview
- ✓ See API reference
- ✓ Links to ARCHITECTURE.md and DEVELOPMENT.md

### 5d. Test API Endpoints Directly

```powershell
# Terminal 2: Test API (with server running)

# Health check
curl http://localhost:5000/api/health

# Get pipeline stages
curl http://localhost:5000/api/pipeline/stages

# Get supported languages
curl http://localhost:5000/api/supported-languages

# Get GNN models
curl http://localhost:5000/api/gnn-models

# Run full pipeline (POST with JSON)
curl -X POST http://localhost:5000/api/pipeline/run `
  -H "Content-Type: application/json" `
  -d '{
    "data_directory": "../Data/1273091433/jeesite",
    "device": "cpu",
    "output_directory": "./outputs"
  }'
```

**Expected:**
```json
{
  "status": "success",
  "message": "Pipeline executed successfully",
  "results": {...}
}
```

---

## Coverage Report

Generate HTML coverage report:

```powershell
pytest tests/ --cov=src --cov-report=html

# Open coverage report
Start-Process "htmlcov/index.html"
```

**Expected coverage:**
- src/cpg/ ≥ 80%
- src/gnn/ ≥ 75%
- src/transformer/ ≥ 70%
- src/xai/ ≥ 70%
- src/web/ ≥ 80% (new)

---

## Performance Testing

### Load Test (Multiple Concurrent Requests)

```powershell
# Install: pip install locust

# Create locustfile.py
cat > locustfile.py << 'EOF'
from locust import HttpUser, between, task

class XRefactorUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(1)
    def health_check(self):
        self.client.get("/api/health")
    
    @task(2)
    def get_stages(self):
        self.client.get("/api/pipeline/stages")
    
    @task(1)
    def get_languages(self):
        self.client.get("/api/supported-languages")
EOF

# Run: locust -f locustfile.py
```

### Benchmark: Pipeline Execution Time

```powershell
# Time full pipeline on various code sizes
Measure-Command {
    python -c "
from src.core.pipeline import XRefactorPipeline
pipeline = XRefactorPipeline('./configs/config.yaml', device='cpu')
pipeline.run_pipeline('../Data/1273091433/jeesite', './outputs')
" | Out-Null
}
```

---

## Continuous Integration (GitHub Actions)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - uses: actions/setup-python@v2
        with:
          python-version: 3.10
      
      - name: Install dependencies
        run: |
          pip install -r xrefactor/requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          cd xrefactor
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Troubleshooting Tests

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'torch'` | Activate venv: `.\.venv\Scripts\Activate.ps1` |
| `GraphSAGEConv not found` | Already fixed; uses `SAGEConv` now |
| `CUDA out of memory` | Use `device='cpu'` or smaller model |
| `Port 5000 already in use` | Kill process or use `--port 5001` |
| `No results in GUI` | Check browser console (F12) for JS errors |

---

## Validation Checklist

Before marking development complete:

- [ ] All unit tests pass: `pytest tests/test_core.py -v`
- [ ] All API tests pass: `pytest tests/test_web.py -v`
- [ ] All integration tests pass: `pytest tests/test_integration.py -v`
- [ ] Backend server starts without errors: `python app.py`
- [ ] Frontend loads in browser: `http://127.0.0.1:5000`
- [ ] Pipeline runs successfully from GUI
- [ ] Results display correctly in GUI
- [ ] Configuration can be saved/loaded
- [ ] API endpoints respond correctly to direct calls
- [ ] Coverage ≥ 80% for core modules

---

## Next Steps

1. **Deploy**: Create Docker container for production deployment
2. **Scale**: Implement distributed processing for large codebases
3. **Extend**: Add support for more languages (Python, C++, JavaScript)
4. **Optimize**: Implement model quantization and caching
5. **Monitor**: Add metrics collection and performance monitoring

---

**Reference:**
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Development: [DEVELOPMENT.md](DEVELOPMENT.md)
- Code: `src/` directory (4 stages)
- Web: `src/web/` + `templates/` + `static/`
