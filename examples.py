"""
Example usage of XRefactor pipeline
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.pipeline import XRefactorPipeline
from src.utils.helpers import LoggerSetup
from loguru import logger


def example_basic_pipeline():
    """Run basic pipeline example"""
    print("\n" + "=" * 80)
    print("XREFACTOR BASIC EXAMPLE")
    print("=" * 80 + "\n")
    
    # Setup logging
    LoggerSetup.setup(log_level="INFO")
    
    # Initialize pipeline
    config_path = "./configs/config.yaml"
    data_dir = "../Data/1273091433/jeesite"  # Sample dataset
    output_dir = "./outputs/example_basic"
    
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = XRefactorPipeline(config_path, device="cuda")
    
    # Run pipeline
    if os.path.isdir(data_dir):
        results = pipeline.run_pipeline(data_dir, output_dir)
        
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Status: {results['status']}")
        print(f"Timestamp: {results['timestamp']}")
        
        for stage, info in results.get('stages', {}).items():
            print(f"{stage}: {info.get('status', 'unknown')}")
    else:
        logger.warning(f"Data directory not found: {data_dir}")


def example_cpg_only():
    """Example: CPG construction only"""
    print("\n" + "=" * 80)
    print("XREFACTOR CPG CONSTRUCTION EXAMPLE")
    print("=" * 80 + "\n")
    
    LoggerSetup.setup(log_level="INFO")
    
    from src.cpg.cpg_builder import CodePropertyGraph
    
    config_path = "./configs/config.yaml"
    data_dir = "../Data/1273091433/jeesite"
    
    if os.path.isdir(data_dir):
        cpg = CodePropertyGraph(language="java")
        cpg.build_from_directory(data_dir)
        
        stats = cpg.get_statistics()
        print("\nCPG Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        logger.warning(f"Data directory not found: {data_dir}")


def example_gnn_only():
    """Example: GNN model creation"""
    print("\n" + "=" * 80)
    print("XREFACTOR GNN MODEL EXAMPLE")
    print("=" * 80 + "\n")
    
    LoggerSetup.setup(log_level="INFO")
    
    import torch
    from src.gnn.gnn_model import HypergraphGNN
    
    # Create model
    model = HypergraphGNN(
        input_dim=1,
        hidden_dims=[256, 256],
        output_dim=128,
        model_type="gat",
        num_heads=8
    )
    
    print(f"GNN Model created successfully!")
    print(f"Model type: GAT")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


def example_transformer_only():
    """Example: Transformer model creation"""
    print("\n" + "=" * 80)
    print("XREFACTOR TRANSFORMER MODEL EXAMPLE")
    print("=" * 80 + "\n")
    
    LoggerSetup.setup(log_level="INFO")
    
    from src.transformer.code_generator import CodeTransformer
    
    # Create model
    model = CodeTransformer(
        model_name="microsoft/codebert-base",
        hidden_size=768,
        num_layers=6
    )
    
    print(f"Transformer Model created successfully!")


if __name__ == "__main__":
    # Run examples
    print("\n" + "=" * 80)
    print("XREFACTOR EXAMPLE SCRIPTS")
    print("=" * 80)
    
    # Uncomment the example you want to run
    
    # example_basic_pipeline()
    example_cpg_only()
    # example_gnn_only()
    # example_transformer_only()
    
    print("\nExamples completed!")
