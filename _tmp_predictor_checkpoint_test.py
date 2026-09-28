from src.core.pipeline import XRefactorPipeline

pipeline = XRefactorPipeline("./configs/config.yaml", device="cpu")
pipeline.config.set("gnn.refactoring_predictor_checkpoint", "./models/refactoring_predictor_trained.pt")

cpg = pipeline.stage_1_cpg_construction("./refactoring_mining/repos/MicroBreweryModel")
gnn_model, node_embeddings = pipeline.stage_2_gnn_reasoning(cpg)
print("node_embeddings shape:", node_embeddings.shape)
print("node_refactoring_logits shape:", pipeline.node_refactoring_logits.shape)
print("node_confidence_scores shape:", pipeline.node_confidence_scores.shape)
print("ALL CHECKS PASSED")
