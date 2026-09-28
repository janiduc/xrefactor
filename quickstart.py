#!/usr/bin/env python
"""
Quick Start Script for XRefactor
Run this to verify installation and test basic functionality
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n" + "=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80 + "\n")
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'torch_geometric': 'PyTorch Geometric',
        'networkx': 'NetworkX',
        'javalang': 'JavaLang',
        'loguru': 'Loguru',
        'yaml': 'PyYAML'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def test_imports():
    """Test importing XRefactor modules"""
    print("\n" + "=" * 80)
    print("TESTING IMPORTS")
    print("=" * 80 + "\n")
    
    try:
        from src.cpg.cpg_builder import CodePropertyGraph
        print("✓ CPG Module")
        
        from src.gnn.gnn_model import HypergraphGNN
        print("✓ GNN Module")
        
        from src.transformer.code_generator import CodeTransformer
        print("✓ Transformer Module")
        
        from src.xai.explanation_module import CausalInferenceModule
        print("✓ XAI Module")
        
        from src.core.pipeline import XRefactorPipeline
        print("✓ Pipeline Module")
        
        from src.utils.helpers import ConfigManager, LoggerSetup
        print("✓ Utility Modules")
        
        print("\n✓ All modules imported successfully!")
        return True
    
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        return False


def test_cpg():
    """Test CPG construction"""
    print("\n" + "=" * 80)
    print("TESTING CPG CONSTRUCTION")
    print("=" * 80 + "\n")
    
    try:
        from src.cpg.cpg_builder import CodePropertyGraph
        
        cpg = CodePropertyGraph(language="java")
        print("✓ CPG initialized")
        
        # Add a test node
        cpg._add_node(
            node_id="test_1",
            node_type="method",
            name="testMethod",
            file_path="/test.java",
            line_number=10,
            code_snippet="void test() {}"
        )
        
        print("✓ Node added to CPG")
        
        stats = cpg.get_statistics()
        print(f"✓ CPG Statistics:")
        print(f"  - Nodes: {stats['total_nodes']}")
        print(f"  - Files: {stats['total_files']}")
        
        return True
    
    except Exception as e:
        print(f"✗ CPG test failed: {e}")
        return False


def test_gnn():
    """Test GNN model"""
    print("\n" + "=" * 80)
    print("TESTING GNN MODEL")
    print("=" * 80 + "\n")
    
    try:
        import torch
        from src.gnn.gnn_model import HypergraphGNN
        
        # Create model
        model = HypergraphGNN(
            input_dim=1,
            hidden_dims=[64],
            output_dim=32,
            model_type="gat"
        )
        print("✓ GNN model created")
        
        # Test forward pass
        model.eval()
        x = torch.randn(5, 1)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        with torch.no_grad():
            node_emb, graph_emb = model(x, edge_index)
        
        print(f"✓ Forward pass successful")
        print(f"  - Node embeddings: {node_emb.shape}")
        print(f"  - Graph embedding: {graph_emb.shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ GNN test failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\n" + "=" * 80)
    print("TESTING CONFIGURATION")
    print("=" * 80 + "\n")
    
    try:
        from src.utils.helpers import ConfigManager
        
        config_path = "./configs/config.yaml"
        if not os.path.exists(config_path):
            print(f"⚠️  Config file not found: {config_path}")
            return False
        
        config = ConfigManager(config_path)
        print("✓ Configuration loaded")
        
        # Test getting values
        language = config.get("cpg.language", "java")
        print(f"✓ CPG language: {language}")
        
        gnn_type = config.get("gnn.model_type", "gat")
        print(f"✓ GNN model type: {gnn_type}")
        
        return True
    
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_dataset():
    """Test dataset access"""
    print("\n" + "=" * 80)
    print("TESTING DATASET ACCESS")
    print("=" * 80 + "\n")
    
    data_dir = "../Data"
    if not os.path.isdir(data_dir):
        print(f"⚠️  Data directory not found: {data_dir}")
        print("   Datasets should be in: ../Data/")
        return False
    
    print(f"✓ Data directory found: {data_dir}")
    
    # Count repositories
    repos = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"✓ Found {len(repos)} repositories")
    
    for repo in repos[:3]:
        repo_path = os.path.join(data_dir, repo)
        files = 0
        for root, dirs, filenames in os.walk(repo_path):
            files += sum(1 for f in filenames if f.endswith(('.java', '.py')))
        print(f"  - {repo}: {files} source files")
    
    return True


def run_quick_test():
    """Run a quick end-to-end test"""
    print("\n" + "=" * 80)
    print("RUNNING QUICK END-TO-END TEST")
    print("=" * 80 + "\n")
    
    try:
        from src.core.pipeline import XRefactorPipeline
        from src.utils.helpers import LoggerSetup
        
        # Setup logging
        LoggerSetup.setup(log_level="INFO")
        
        # Initialize pipeline
        config_path = "./configs/config.yaml"
        pipeline = XRefactorPipeline(config_path, device="cpu")  # Use CPU for testing
        print("✓ Pipeline initialized")
        
        # Stage 1: CPG
        stage1 = pipeline.stage_1_cpg_construction("../Data/1273091433/jeesite")
        if stage1:
            print("✓ Stage 1 (CPG Construction) completed")
        else:
            print("⚠️  Stage 1 skipped (data not available)")
        
        print("\n✓ Quick test completed!")
        return True
    
    except Exception as e:
        print(f"⚠️  Quick test incomplete: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="XRefactor Quick Start")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite (including end-to-end)"
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Only test dataset access"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("XREFACTOR QUICK START")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    # Core tests
    if check_dependencies():
        passed += 1
    else:
        failed += 1
        print("\n❌ Dependency check failed. Please install dependencies:")
        print("   pip install -r requirements.txt")
        return
    
    if test_imports():
        passed += 1
    else:
        failed += 1
    
    if test_config():
        passed += 1
    else:
        failed += 1
    
    if test_dataset():
        passed += 1
    else:
        failed += 1
    
    # Optional tests
    if not args.dataset:
        if test_cpg():
            passed += 1
        else:
            failed += 1
        
        if test_gnn():
            passed += 1
        else:
            failed += 1
        
        if args.full:
            if run_quick_test():
                passed += 1
            else:
                failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All tests passed! XRefactor is ready to use.")
        print("\nNext steps:")
        print("1. Review configs/config.yaml")
        print("2. Run: python main.py --data-dir ../Data --output ./outputs")
        print("3. See README.md for detailed usage")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please address issues above.")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
