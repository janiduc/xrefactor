"""
Pytest configuration and fixtures
Adds xrefactor directory to sys.path for proper imports
"""

import sys
import os
from pathlib import Path

# Add xrefactor directory to Python path
xrefactor_dir = Path(__file__).parent.parent
sys.path.insert(0, str(xrefactor_dir))

# Verify imports work
try:
    from src.core.pipeline import XRefactorPipeline
    from src.cpg.cpg_builder import CodePropertyGraph
    from src.gnn.gnn_model import HypergraphGNN
    # Plain ASCII: this module is imported before pytest installs its capture,
    # so on a cp1252 console (Windows default) non-ASCII here raises
    # UnicodeEncodeError and aborts collection whenever -s is used.
    print("Core imports verified")
except ImportError as e:
    print(f"Import warning: {e}")
