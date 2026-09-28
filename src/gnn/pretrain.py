"""
Self-supervised pretraining for the GNN stage.

No labeled refactoring dataset exists, so instead of training the classifier
heads on invented labels, this trains the node encoder + HypergraphGNN on a
link-prediction task: given two node embeddings, predict whether a CPG edge
exists between them. This is a standard self-supervised signal (similar to a
Graph Autoencoder) that shapes embeddings around real graph structure instead
of leaving them randomly initialized.

Usage:
    python -m src.gnn.pretrain --data-dirs ../Data/1273091433/jeesite --epochs 20
"""

import argparse
import glob
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch_geometric.utils import negative_sampling

from ..cpg.cpg_builder import CodePropertyGraph
from .gnn_model import HypergraphGNN, DependencyHypergraphEncoder, convert_cpg_to_geometric_data


def _build_geometric_datasets(data_dirs: List[str], language: str,
                               node_type_to_id: dict, encoder: DependencyHypergraphEncoder):
    """Build one PyG Data object per source directory"""
    datasets = []
    for data_dir in data_dirs:
        cpg = CodePropertyGraph(language=language)
        cpg.build_from_directory(data_dir)
        if len(cpg.nodes) < 2 or len(cpg.edges) == 0:
            logger.warning(f"Skipping {data_dir}: not enough nodes/edges to train on")
            continue
        cpg_dict = cpg.to_dict()
        data = convert_cpg_to_geometric_data(cpg_dict, node_type_to_id, encoder=encoder)
        datasets.append(data)
        logger.info(f"Loaded {data_dir}: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    return datasets


def pretrain_link_prediction(data_dirs: List[str],
                              node_type_to_id: dict,
                              language: str = "java",
                              hidden_dims: List[int] = None,
                              output_dim: int = 128,
                              node_embedding_dim: int = 64,
                              model_type: str = "gat",
                              num_heads: int = 8,
                              epochs: int = 20,
                              lr: float = 0.001,
                              device: str = "cpu") -> dict:
    """
    Train DependencyHypergraphEncoder + HypergraphGNN on link prediction.
    
    Returns:
        Dict with "node_encoder" and "gnn_model" state dicts, ready to be saved
        as a checkpoint via torch.save().
    """
    hidden_dims = hidden_dims or [256, 256]
    
    node_encoder = DependencyHypergraphEncoder(
        node_type_vocab_size=len(node_type_to_id), embedding_dim=node_embedding_dim
    ).to(device)
    gnn_model = HypergraphGNN(
        input_dim=node_embedding_dim, hidden_dims=hidden_dims, output_dim=output_dim,
        model_type=model_type, num_heads=num_heads
    ).to(device)
    
    datasets = _build_geometric_datasets(data_dirs, language, node_type_to_id, node_encoder)
    if not datasets:
        raise ValueError("No usable CPGs were built from the given data directories")
    
    params = list(node_encoder.parameters()) + list(gnn_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    
    node_encoder.train()
    gnn_model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for data in datasets:
            x = data.x.float().to(device)
            edge_index = data.edge_index.to(device)
            if edge_index.numel() == 0:
                continue
            
            optimizer.zero_grad()
            
            node_embeddings, _ = gnn_model(x, edge_index)
            
            neg_edge_index = negative_sampling(
                edge_index, num_nodes=x.shape[0], num_neg_samples=edge_index.shape[1]
            )
            
            pos_scores = (node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]).sum(dim=-1)
            neg_scores = (node_embeddings[neg_edge_index[0]] * node_embeddings[neg_edge_index[1]]).sum(dim=-1)
            
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        logger.info(f"[pretrain] epoch {epoch}/{epochs} - link prediction loss: {epoch_loss / len(datasets):.4f}")
    
    return {
        "node_encoder_state": node_encoder.state_dict(),
        "gnn_model_state": gnn_model.state_dict(),
        "config": {
            "node_embedding_dim": node_embedding_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "model_type": model_type,
            "num_heads": num_heads
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Self-supervised GNN pretraining (link prediction)")
    parser.add_argument("--data-dirs", nargs="+", required=True, help="One or more source directories/glob patterns to train on")
    parser.add_argument("--language", default="java")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="./models/gnn_pretrained.pt")
    args = parser.parse_args()
    
    resolved_dirs = []
    for pattern in args.data_dirs:
        matches = glob.glob(pattern)
        resolved_dirs.extend(matches if matches else [pattern])
    
    node_type_to_id = {
        "class": 0, "method": 1, "function": 2, "field": 3,
        "variable": 4, "statement": 5, "call": 6, "definition": 7
    }
    
    checkpoint = pretrain_link_prediction(
        resolved_dirs, node_type_to_id, language=args.language,
        epochs=args.epochs, lr=args.lr, device=args.device
    )
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(checkpoint, args.output)
    logger.info(f"Saved pretrained GNN checkpoint to {args.output}")


if __name__ == "__main__":
    main()
