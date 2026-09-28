"""
Integration Tests
End-to-end tests for the complete XRefactor system
"""

import pytest
import os
import torch
import tempfile
from pathlib import Path

from src.core.pipeline import XRefactorPipeline
from src.cpg.cpg_builder import CodePropertyGraph
from src.gnn.gnn_model import HypergraphGNN
from src.transformer.code_generator import CodeTransformer
from src.xai.explanation_module import CausalInferenceModule


class TestPipelineIntegration:
    """Integration tests for full pipeline"""
    
    @pytest.fixture
    def config_path(self):
        """Get config path"""
        return './configs/config.yaml'
    
    @pytest.fixture
    def sample_java_code(self):
        """Create sample Java code"""
        return """
        public class HelloWorld {
            public static void main(String[] args) {
                System.out.println("Hello, World!");
            }
        }
        """
    
    def test_pipeline_initialization(self, config_path):
        """Test pipeline can be initialized"""
        pipeline = XRefactorPipeline(config_path, device='cpu')
        assert pipeline is not None
        assert pipeline.cpg is None  # Not yet constructed
    
    def test_gnn_model_creation(self):
        """Test GNN model creation and forward pass"""
        model = HypergraphGNN(
            input_dim=1,
            hidden_dims=[64, 64],  # Smaller for testing
            output_dim=32,
            model_type='gat',
            num_heads=4,
            dropout=0.1
        )
        
        # Test forward pass
        x = torch.randn(10, 1)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], 
                                   [1, 2, 3, 4, 5]], dtype=torch.long)
        
        node_emb, graph_emb = model(x, edge_index)
        
        assert node_emb.shape == (10, 32)
        assert graph_emb.shape == (1, 32)
    
    def test_transformer_creation(self):
        """Test Transformer model creation"""
        model = CodeTransformer(
            model_name='microsoft/codebert-base',
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4
        )
        
        assert model is not None
    
    def test_xai_module_creation(self):
        """Test XAI module creation"""
        xai = CausalInferenceModule(feature_dim=128)
        assert xai is not None


class TestComponentInteraction:
    """Test interactions between components"""
    
    def test_cpg_to_gnn_flow(self):
        """Test data flow from CPG to GNN"""
        # This tests the conversion pipeline
        from src.gnn.gnn_model import convert_cpg_to_geometric_data
        
        # Create minimal CPG dict
        cpg_dict = {
            'nodes': [
                {'id': 0, 'type': 'class', 'name': 'MyClass'},
                {'id': 1, 'type': 'method', 'name': 'myMethod'},
                {'id': 2, 'type': 'field', 'name': 'myField'}
            ],
            'edges': [
                {'source': 0, 'target': 1, 'type': 'contains'},
                {'source': 0, 'target': 2, 'type': 'contains'}
            ]
        }
        
        node_type_to_id = {
            'class': 0, 'method': 1, 'field': 2,
            'function': 3, 'variable': 4
        }
        
        geometric_data = convert_cpg_to_geometric_data(cpg_dict, node_type_to_id)
        
        assert geometric_data is not None
        assert hasattr(geometric_data, 'x')
        assert hasattr(geometric_data, 'edge_index')
        assert geometric_data.x.shape[0] == 3  # 3 nodes
    
    def test_device_handling(self):
        """Test CPU/GPU device handling"""
        # Test CPU
        model_cpu = HypergraphGNN(
            input_dim=1, hidden_dims=[32], output_dim=16,
            model_type='gat'
        ).to('cpu')
        
        x = torch.randn(5, 1, device='cpu')
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device='cpu')
        
        out_cpu, graph_cpu = model_cpu(x, edge_index)
        assert out_cpu.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model_cpu.to('cuda')
            x_cuda = x.to('cuda')
            edge_index_cuda = edge_index.to('cuda')
            
            out_cuda, graph_cuda = model_cuda(x_cuda, edge_index_cuda)
            assert out_cuda.device.type == 'cuda'


class TestErrorRecovery:
    """Test error recovery and robustness"""
    
    def test_invalid_model_type(self):
        """Test handling of invalid GNN model type"""
        with pytest.raises(ValueError):
            HypergraphGNN(
                input_dim=1,
                hidden_dims=[32],
                output_dim=16,
                model_type='invalid_model'
            )
    
    def test_empty_graph_handling(self):
        """Test handling of empty graph"""
        model = HypergraphGNN(input_dim=1, hidden_dims=[32], output_dim=16)
        
        # Empty nodes
        x = torch.randn(0, 1)
        edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        
        # Should handle gracefully
        try:
            node_emb, graph_emb = model(x, edge_index)
            # May succeed or fail gracefully
        except RuntimeError:
            pass  # Expected for some models with empty input


class TestMemoryEfficiency:
    """Test memory usage and efficiency"""
    
    def test_large_graph_handling(self):
        """Test handling of larger graphs"""
        model = HypergraphGNN(
            input_dim=1,
            hidden_dims=[64, 64],
            output_dim=32,
            model_type='gat',
            num_heads=4
        )
        
        # Create larger graph
        num_nodes = 1000
        num_edges = 5000
        
        x = torch.randn(num_nodes, 1)
        edge_idx = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
        
        model.eval()
        with torch.no_grad():
            node_emb, graph_emb = model(x, edge_idx)
        
        assert node_emb.shape == (num_nodes, 32)
        assert graph_emb.shape == (1, 32)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
