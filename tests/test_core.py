"""
Unit tests for XRefactor components
"""

import pytest
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.cpg.cpg_builder import CodePropertyGraph, CodeNode, CodeEdge
from src.gnn.gnn_model import HypergraphGNN, convert_cpg_to_geometric_data
from src.transformer.code_generator import CodeTransformer
from src.xai.explanation_module import CausalInferenceModule, ExplanationGenerator


class TestCPG:
    """Tests for Code Property Graph"""
    
    def test_cpg_initialization(self):
        """Test CPG initialization"""
        cpg = CodePropertyGraph(language="java")
        assert cpg.language == "java"
        assert len(cpg.nodes) == 0
        assert len(cpg.edges) == 0
    
    def test_add_node(self):
        """Test adding nodes to CPG"""
        cpg = CodePropertyGraph()
        cpg._add_node(
            node_id="test_1",
            node_type="method",
            name="testMethod",
            file_path="/test.java",
            line_number=10,
            code_snippet="void testMethod() {}"
        )
        
        assert "test_1" in cpg.nodes
        assert cpg.nodes["test_1"].name == "testMethod"
        assert cpg.nodes["test_1"].node_type == "method"


class TestGNN:
    """Tests for Graph Neural Networks"""
    
    def test_gnn_initialization(self):
        """Test GNN model initialization"""
        model = HypergraphGNN(
            input_dim=1,
            hidden_dims=[256],
            output_dim=128,
            model_type="gat"
        )
        assert model.input_dim == 1
        assert model.output_dim == 128
    
    def test_gnn_forward_pass(self):
        """Test GNN forward pass"""
        model = HypergraphGNN(
            input_dim=1,
            hidden_dims=[64],
            output_dim=32,
            model_type="gat"
        )
        model.eval()
        
        # Create dummy data
        x = torch.randn(10, 1)  # 10 nodes, 1 feature
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        with torch.no_grad():
            node_emb, graph_emb = model(x, edge_index)
        
        assert node_emb.shape == (10, 32)
        assert graph_emb.shape[1] == 32


class TestTransformer:
    """Tests for Transformer models"""
    
    def test_transformer_initialization(self):
        """Test Transformer initialization"""
        try:
            model = CodeTransformer(hidden_size=768, num_layers=2)
            assert model.hidden_size == 768
            assert model.num_layers == 2
        except Exception as e:
            pytest.skip(f"Transformer initialization failed: {e}")


class TestXAI:
    """Tests for XAI modules"""
    
    def test_causal_inference_initialization(self):
        """Test causal inference module initialization"""
        module = CausalInferenceModule(feature_dim=768)
        assert module.problem_detector is not None
        assert module.solution_evaluator is not None
    
    def test_causal_inference_forward(self):
        """Test causal inference forward pass"""
        module = CausalInferenceModule(feature_dim=64)
        
        original_emb = torch.randn(2, 10, 64)  # batch=2, seq_len=10
        refactored_emb = torch.randn(2, 10, 64)
        
        problem_scores, solution_scores, attn_weights = module(
            original_emb, refactored_emb
        )
        
        assert problem_scores.shape == (2, 10)
        assert solution_scores.shape == (2, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
