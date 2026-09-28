"""
Tests that each of the 4 _load_*_checkpoint_if_available paths (GNN,
refactoring predictor, transformer, XAI) actually change model weights when
given a real (tiny, synthetic) non-null checkpoint - closing the gap where
checkpoint loading was previously only ever exercised with the null/missing
path, never with an actually-loaded checkpoint.
"""

import torch
import torch.nn as nn

from src.core.pipeline import XRefactorPipeline
from src.cpg.cpg_builder import CodeEdge, CodePropertyGraph
from src.gnn.gnn_model import DependencyHypergraphEncoder, HypergraphGNN, RefactoringPredictor
from src.transformer.code_generator import CodeTransformer
from src.xai.explanation_module import CausalInferenceModule


def _make_pipeline():
    return XRefactorPipeline(config_path="./configs/config.yaml", device="cpu")


def _fill(module: nn.Module, value: float):
    with torch.no_grad():
        for p in module.parameters():
            p.fill_(value)


def _assert_filled(module: nn.Module, value: float):
    param = next(module.parameters())
    assert torch.allclose(param, torch.full_like(param, value), atol=1e-4)


class TestGNNCheckpointLoading:
    def test_loading_changes_weights(self, tmp_path):
        pipeline = _make_pipeline()
        pipeline.node_encoder = DependencyHypergraphEncoder(node_type_vocab_size=8, embedding_dim=64)
        pipeline.gnn_model = HypergraphGNN(input_dim=64, hidden_dims=[256, 256], output_dim=128, model_type="gat")

        distinctive_encoder = DependencyHypergraphEncoder(node_type_vocab_size=8, embedding_dim=64)
        distinctive_gnn = HypergraphGNN(input_dim=64, hidden_dims=[256, 256], output_dim=128, model_type="gat")
        _fill(distinctive_encoder, 0.1234)
        _fill(distinctive_gnn, 0.5678)

        checkpoint_path = tmp_path / "gnn_test.pt"
        torch.save({"node_encoder_state": distinctive_encoder.state_dict(),
                    "gnn_model_state": distinctive_gnn.state_dict()}, checkpoint_path)

        pipeline.config.set("gnn.pretrained_checkpoint", str(checkpoint_path))
        pipeline._load_gnn_checkpoint_if_available()

        _assert_filled(pipeline.node_encoder, 0.1234)
        _assert_filled(pipeline.gnn_model, 0.5678)

    def test_null_checkpoint_leaves_weights_untouched(self):
        pipeline = _make_pipeline()
        pipeline.node_encoder = DependencyHypergraphEncoder(node_type_vocab_size=8, embedding_dim=64)
        pipeline.gnn_model = HypergraphGNN(input_dim=64, hidden_dims=[256, 256], output_dim=128, model_type="gat")
        _fill(pipeline.node_encoder, 0.9999)

        pipeline.config.set("gnn.pretrained_checkpoint", None)
        pipeline._load_gnn_checkpoint_if_available()

        _assert_filled(pipeline.node_encoder, 0.9999)


class TestRefactoringPredictorCheckpointLoading:
    def test_loading_changes_weights(self, tmp_path):
        pipeline = _make_pipeline()
        gnn = HypergraphGNN(input_dim=64, hidden_dims=[256, 256], output_dim=128, model_type="gat")
        pipeline.refactoring_predictor = RefactoringPredictor(gnn, output_dim=128, num_refactoring_types=10)

        distinctive = RefactoringPredictor(gnn, output_dim=128, num_refactoring_types=10)
        _fill(distinctive.classifier, 0.2222)
        _fill(distinctive.confidence_predictor, 0.3333)

        checkpoint_path = tmp_path / "predictor_test.pt"
        torch.save({"classifier_state": distinctive.classifier.state_dict(),
                    "confidence_predictor_state": distinctive.confidence_predictor.state_dict()}, checkpoint_path)

        pipeline.config.set("gnn.refactoring_predictor_checkpoint", str(checkpoint_path))
        pipeline._load_refactoring_predictor_checkpoint_if_available()

        _assert_filled(pipeline.refactoring_predictor.classifier, 0.2222)
        _assert_filled(pipeline.refactoring_predictor.confidence_predictor, 0.3333)


class TestTransformerCheckpointLoading:
    def test_loading_changes_weights(self, tmp_path):
        pipeline = _make_pipeline()
        pipeline.transformer = CodeTransformer(model_name="microsoft/codebert-base",
                                                gnn_embedding_dim=128, hidden_size=768, num_layers=2)

        distinctive = CodeTransformer(model_name="microsoft/codebert-base",
                                       gnn_embedding_dim=128, hidden_size=768, num_layers=2)
        _fill(distinctive.decoder, 0.4444)
        _fill(distinctive.fusion_layer, 0.5555)
        _fill(distinctive.gnn_projection, 0.6666)

        checkpoint_path = tmp_path / "transformer_test.pt"
        torch.save({"decoder_state": distinctive.decoder.state_dict(),
                    "fusion_layer_state": distinctive.fusion_layer.state_dict(),
                    "gnn_projection_state": distinctive.gnn_projection.state_dict()}, checkpoint_path)

        pipeline.config.set("transformer.pretrained_checkpoint", str(checkpoint_path))
        pipeline._load_transformer_checkpoint_if_available()

        _assert_filled(pipeline.transformer.decoder, 0.4444)
        _assert_filled(pipeline.transformer.fusion_layer, 0.5555)
        _assert_filled(pipeline.transformer.gnn_projection, 0.6666)


class TestXAICheckpointLoading:
    def test_loading_changes_weights(self, tmp_path):
        pipeline = _make_pipeline()
        pipeline.xai_module = CausalInferenceModule(feature_dim=128)

        distinctive = CausalInferenceModule(feature_dim=128)
        _fill(distinctive.problem_detector, 0.1111)
        _fill(distinctive.solution_evaluator, 0.7777)

        checkpoint_path = tmp_path / "xai_test.pt"
        torch.save({"problem_detector_state": distinctive.problem_detector.state_dict(),
                    "solution_evaluator_state": distinctive.solution_evaluator.state_dict()}, checkpoint_path)

        pipeline.config.set("xai.pretrained_checkpoint", str(checkpoint_path))
        pipeline._load_xai_checkpoint_if_available()

        _assert_filled(pipeline.xai_module.problem_detector, 0.1111)
        _assert_filled(pipeline.xai_module.solution_evaluator, 0.7777)


def _build_tiny_cpg() -> CodePropertyGraph:
    """A minimal synthetic CPG: two methods, one calling the other, plus a field."""
    cpg = CodePropertyGraph(language="java")
    cpg._add_node(node_id="class_A.java_Foo", node_type="class", name="Foo",
                  file_path="A.java", line_number=1, code_snippet="class Foo {")
    cpg._add_node(node_id="method_A.java_bar", node_type="method", name="bar",
                  file_path="A.java", line_number=2, code_snippet="void bar() { baz(); }")
    cpg._add_node(node_id="method_A.java_baz", node_type="method", name="baz",
                  file_path="A.java", line_number=3, code_snippet="void baz() {}")
    cpg._add_node(node_id="field_A.java_count", node_type="field", name="count",
                  file_path="A.java", line_number=4, code_snippet="int count;")
    cpg.edges.append(CodeEdge(source_id="method_A.java_bar", target_id="method_A.java_baz", edge_type="calls"))
    cpg._build_cross_file_dependencies()
    return cpg


class TestPipelineIntegrationWithCheckpoints:
    def test_checkpoints_change_stage_2_output(self, tmp_path):
        """Running stage 2 with a real (tiny, distinctive) GNN checkpoint should
        produce different node embeddings than running it with no checkpoint at
        all - a concrete, cheap regression guard that checkpoint wiring actually
        takes effect end-to-end, not just that the loader function runs."""
        cpg = _build_tiny_cpg()

        pipeline_random = _make_pipeline()
        pipeline_random.config.set("gnn.pretrained_checkpoint", None)
        _, embeddings_random = pipeline_random.stage_2_gnn_reasoning(cpg)

        node_encoder = DependencyHypergraphEncoder(node_type_vocab_size=8, embedding_dim=64)
        gnn_model = HypergraphGNN(input_dim=64, hidden_dims=[256, 256], output_dim=128, model_type="gat")
        _fill(node_encoder, 0.42)
        _fill(gnn_model, 0.24)
        checkpoint_path = tmp_path / "gnn_integration_test.pt"
        torch.save({"node_encoder_state": node_encoder.state_dict(),
                    "gnn_model_state": gnn_model.state_dict()}, checkpoint_path)

        pipeline_checkpointed = _make_pipeline()
        pipeline_checkpointed.config.set("gnn.pretrained_checkpoint", str(checkpoint_path))
        _, embeddings_checkpointed = pipeline_checkpointed.stage_2_gnn_reasoning(cpg)

        assert not torch.equal(embeddings_random, embeddings_checkpointed)
