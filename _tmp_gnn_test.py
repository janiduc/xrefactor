import torch
from src.gnn.gnn_model import (
    HypergraphGNN, DependencyHypergraphEncoder, RefactoringPredictor, convert_cpg_to_geometric_data
)

cpg_dict = {
    "nodes": {
        "n0": {"type": "class", "name": "Foo"},
        "n1": {"type": "method", "name": "compute"},
        "n2": {"type": "method", "name": "helper"},
        "n3": {"type": "variable", "name": "sum"},
        "n4": {"type": "statement", "name": "IfStatement"},
    },
    "edges": [
        {"source": "n0", "target": "n1", "type": "calls"},
        {"source": "n1", "target": "n2", "type": "calls"},
        {"source": "n1", "target": "n3", "type": "data_flow"},
        {"source": "n1", "target": "n4", "type": "control_flow"},
    ]
}
node_type_to_id = {"class": 0, "method": 1, "function": 2, "field": 3, "variable": 4, "statement": 5, "call": 6, "definition": 7}

encoder = DependencyHypergraphEncoder(node_type_vocab_size=len(node_type_to_id), embedding_dim=16)
data = convert_cpg_to_geometric_data(cpg_dict, node_type_to_id, encoder=encoder)
print("x shape (encoder):", data.x.shape)
assert data.x.shape == (5, 16)
assert not data.x.requires_grad, "encoded features should be detached"

data_no_encoder = convert_cpg_to_geometric_data(cpg_dict, node_type_to_id)
print("x shape (one-hot fallback):", data_no_encoder.x.shape)
assert data_no_encoder.x.shape == (5, 8)
# one-hot rows should not be an ordinal scalar - check row for 'method' (id=1) has exactly one 1.0
assert data_no_encoder.x[1].sum().item() == 1.0
assert data_no_encoder.x[1][1].item() == 1.0

gnn = HypergraphGNN(input_dim=16, hidden_dims=[32, 32], output_dim=8, model_type="gat", num_heads=4)
predictor = RefactoringPredictor(gnn, output_dim=8, num_refactoring_types=10)

gnn.eval(); predictor.eval()
with torch.no_grad():
    node_emb, graph_emb = gnn(data.x, data.edge_index)
    logits, confidence = predictor(data.x, data.edge_index)

print("node_emb shape:", node_emb.shape)
print("graph_emb shape:", graph_emb.shape)
print("logits shape:", logits.shape)
print("confidence shape:", confidence.shape)

assert node_emb.shape == (5, 8)
assert logits.shape == (5, 10), "RefactoringPredictor should predict per-node, not per-graph"
assert confidence.shape == (5, 1)

# Confirm GAT hidden width no longer explodes to out_dim * num_heads across layers
first_layer = gnn.layers[0]
print("first GAT layer out_channels (per head):", first_layer.out_channels, "heads:", first_layer.heads)
assert first_layer.out_channels * first_layer.heads <= 40  # ~32, not 32*4=128

print("ALL CHECKS PASSED")
