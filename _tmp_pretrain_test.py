import os, torch
from src.gnn.pretrain import pretrain_link_prediction
from src.core.pipeline import XRefactorPipeline

node_type_to_id = {
    "class": 0, "method": 1, "function": 2, "field": 3,
    "variable": 4, "statement": 5, "call": 6, "definition": 7
}

data_dir = "../Data/ErikHage/MicroBreweryModel"
assert os.path.isdir(data_dir), f"missing {data_dir}"

checkpoint = pretrain_link_prediction(
    [data_dir], node_type_to_id, language="java",
    hidden_dims=[32, 32], output_dim=16, node_embedding_dim=16,
    epochs=2, lr=0.01, device="cpu"
)
os.makedirs("./_tmp_models", exist_ok=True)
ckpt_path = "./_tmp_models/gnn_pretrained_test.pt"
torch.save(checkpoint, ckpt_path)
print("checkpoint saved:", ckpt_path)

# Now verify pipeline can load it without shape mismatches when configs match
pipeline = XRefactorPipeline("./configs/config.yaml", device="cpu")
pipeline.config.set("gnn.hidden_dims", [32, 32])
pipeline.config.set("gnn.output_dim", 16)
pipeline.config.set("gnn.node_embedding_dim", 16)
pipeline.config.set("gnn.pretrained_checkpoint", ckpt_path)

cpg = pipeline.stage_1_cpg_construction(data_dir)
gnn_model, node_embeddings = pipeline.stage_2_gnn_reasoning(cpg)
print("node_embeddings shape:", node_embeddings.shape)
print("ALL CHECKS PASSED")
