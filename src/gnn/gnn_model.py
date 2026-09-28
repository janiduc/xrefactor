"""
Stage 2: Graph Neural Network (GNN) for Structural Reasoning
Uses GNN to reason over dependency hypergraphs and model multi-entity interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from loguru import logger
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class HypergraphGNN(nn.Module):
    """
    Graph Neural Network for reasoning over dependency hypergraphs.
    Supports multiple GNN architectures: GAT, GCN, GraphSAGE.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 model_type: str = "gat",
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_layers: int = 3,
                 activation: str = "relu"):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            model_type: "gat", "gcn", or "graphsage"
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            num_layers: Number of layers
            activation: Activation function
        """
        super(HypergraphGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.model_type = model_type
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.activation = activation
        
        logger.info(f"Initializing {model_type.upper()} with dims: {input_dim} -> {hidden_dims} -> {output_dim}")
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Build layers with correct dimensionality for GAT concat behavior
        hidden_layers = hidden_dims + [output_dim]
        current_dim = input_dim
        for i, out_dim in enumerate(hidden_layers):
            is_last_layer = i == len(hidden_layers) - 1
            if model_type == "gat":
                concat = not is_last_layer
                # Divide by num_heads so concatenated width matches the configured
                # hidden dim instead of exploding to out_dim * num_heads per layer
                per_head_dim = max(1, out_dim // num_heads) if concat else out_dim
                self.layers.append(
                    GATConv(current_dim, per_head_dim, heads=num_heads, dropout=dropout, concat=concat)
                )
                next_dim = per_head_dim * num_heads if concat else out_dim
            elif model_type == "gcn":
                self.layers.append(GCNConv(current_dim, out_dim))
                next_dim = out_dim
            elif model_type == "graphsage":
                self.layers.append(SAGEConv(current_dim, out_dim))
                next_dim = out_dim
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            if not is_last_layer:  # No batch norm on output layer
                self.batch_norms.append(nn.BatchNorm1d(next_dim))
            current_dim = next_dim
        
        self.activation_fn = self._get_activation(activation)
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        if activation == "relu":
            return F.relu
        elif activation == "elu":
            return F.elu
        elif activation == "gelu":
            return F.gelu
        else:
            return F.relu
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for graph-level pooling
        
        Returns:
            node_embeddings: [num_nodes, output_dim]
            graph_embedding: [batch_size, output_dim] (if batch provided)
        """
        node_embedding = x
        
        for i, layer in enumerate(self.layers):
            node_embedding = layer(node_embedding, edge_index)
            
            if i < len(self.batch_norms):
                node_embedding = self.batch_norms[i](node_embedding)
                node_embedding = self.activation_fn(node_embedding)
                node_embedding = F.dropout(node_embedding, p=self.dropout, training=self.training)
        
        # Graph-level embedding via global pooling
        if batch is not None:
            graph_embedding = global_mean_pool(node_embedding, batch)
        else:
            graph_embedding = node_embedding.mean(dim=0, keepdim=True)
        
        return node_embedding, graph_embedding


class DependencyHypergraphEncoder(nn.Module):
    """
    Encodes CPG into node features for GNN processing
    """
    
    def __init__(self, node_type_vocab_size: int = 50, embedding_dim: int = 64):
        super(DependencyHypergraphEncoder, self).__init__()
        
        self.node_type_embedding = nn.Embedding(node_type_vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
    
    def encode_nodes(self, nodes: List[Dict[str, Any]], node_type_to_id: Dict[str, int]) -> torch.Tensor:
        """
        Encode CPG nodes into feature vectors
        
        Args:
            nodes: List of node dictionaries from CPG
            node_type_to_id: Mapping of node type to ID
        
        Returns:
            Node features tensor [num_nodes, embedding_dim]
        """
        type_ids = [node_type_to_id.get(node.get("type", "unknown"), 0) for node in nodes]
        type_ids = torch.tensor(type_ids, dtype=torch.long)
        
        node_features = self.node_type_embedding(type_ids)
        
        return node_features


class RefactoringPredictor(nn.Module):
    """
    Predicts refactoring recommendations based on GNN embeddings
    """
    
    def __init__(self, gnn: HypergraphGNN, output_dim: int = 128, 
                 num_refactoring_types: int = 10):
        super(RefactoringPredictor, self).__init__()
        
        self.gnn = gnn
        self.classifier = nn.Sequential(
            nn.Linear(gnn.output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_refactoring_types)
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(gnn.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict a refactoring recommendation and confidence score per node, so each
        CPG entity (method/class/etc.) gets its own suggestion instead of one
        prediction for the whole graph.
        
        Returns:
            refactoring_logits: [num_nodes, num_refactoring_types]
            confidence_scores: [num_nodes, 1]
        """
        node_embeddings, _ = self.gnn(x, edge_index, batch)
        
        refactoring_logits = self.classifier(node_embeddings)
        confidence_scores = self.confidence_predictor(node_embeddings)
        
        return refactoring_logits, confidence_scores


class GNNTrainer:
    """
    Trainer for GNN models
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda", learning_rate: float = 0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized GNN trainer on device: {device}")
    
    def train_step(self, batch_data: Data) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        x = batch_data.x.to(self.device)
        edge_index = batch_data.edge_index.to(self.device)
        y = batch_data.y.to(self.device) if hasattr(batch_data, 'y') else None
        batch = batch_data.batch.to(self.device) if hasattr(batch_data, 'batch') else None
        
        logits, _ = self.model(x, edge_index, batch)
        
        if y is not None:
            loss = self.loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return loss.item()
        
        return 0.0
    
    def eval_step(self, batch_data: Data) -> float:
        """Single evaluation step"""
        self.model.eval()
        
        with torch.no_grad():
            x = batch_data.x.to(self.device)
            edge_index = batch_data.edge_index.to(self.device)
            y = batch_data.y.to(self.device) if hasattr(batch_data, 'y') else None
            batch = batch_data.batch.to(self.device) if hasattr(batch_data, 'batch') else None
            
            logits, _ = self.model(x, edge_index, batch)
            
            if y is not None:
                loss = self.loss_fn(logits, y)
                return loss.item()
        
        return 0.0


def convert_cpg_to_geometric_data(cpg_dict: Dict[str, Any], 
                                   node_type_to_id: Dict[str, int],
                                   encoder: Optional[DependencyHypergraphEncoder] = None) -> Data:
    """
    Convert CPG dictionary to PyG Data object
    
    Args:
        cpg_dict: CPG as dictionary (from cpg_builder.to_dict())
        node_type_to_id: Mapping of node type strings to IDs
        encoder: Optional DependencyHypergraphEncoder used to build learned node-type
            embeddings. Without it, node types are one-hot encoded so the GNN still
            receives a categorical (non-ordinal) signal rather than a raw type index.
    
    Returns:
        PyG Data object
    """
    nodes = cpg_dict["nodes"]
    edges = cpg_dict["edges"]
    
    # Handle both dict and list formats for nodes
    if isinstance(nodes, list):
        # Convert list format to dict format
        nodes_dict = {}
        node_id_map = {}
        for idx, node_info in enumerate(nodes):
            node_id = node_info.get("id", idx)
            nodes_dict[node_id] = node_info
            node_id_map[node_id] = idx
        nodes = nodes_dict
    else:
        # Create node ID mapping from dict format
        node_id_map = {node_id: idx for idx, node_id in enumerate(nodes.keys())}
    
    ordered_node_ids = sorted(nodes.keys(), key=lambda k: node_id_map.get(k, 0))
    ordered_nodes = [nodes[node_id] for node_id in ordered_node_ids]
    
    # Create node features
    if encoder is not None:
        x = encoder.encode_nodes(ordered_nodes, node_type_to_id).detach()
    else:
        vocab_size = max(node_type_to_id.values(), default=0) + 1
        node_features = []
        for node_info in ordered_nodes:
            node_type_id = node_type_to_id.get(node_info["type"], 0)
            one_hot = [0.0] * vocab_size
            one_hot[node_type_id] = 1.0
            node_features.append(one_hot)
        x = torch.tensor(node_features, dtype=torch.float32)
    
    # Create edge index
    edge_index_list = [[], []]
    for edge in edges:
        src_id = edge.get("source", edge.get("src"))
        tgt_id = edge.get("target", edge.get("tgt"))
        src_idx = node_id_map.get(src_id)
        tgt_idx = node_id_map.get(tgt_id)
        if src_idx is not None and tgt_idx is not None:
            edge_index_list[0].append(src_idx)
            edge_index_list[1].append(tgt_idx)
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long) if edge_index_list[0] else torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)
