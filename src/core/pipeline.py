"""
Core XRefactor Pipeline Orchestrator
Coordinates all stages: CPG construction, GNN reasoning, Transformer generation, XAI explanation
"""

import torch
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import json
from datetime import datetime

from ..cpg.cpg_builder import CodePropertyGraph
from ..cpg.smell_detector import SmellDetector
from ..refactor.engine import SUPPORTED_PATTERNS, attempt_refactoring, find_applicable_pattern
from ..transformer.pattern_validators import validate as validate_pattern
from ..gnn.gnn_model import HypergraphGNN, DependencyHypergraphEncoder, RefactoringPredictor, convert_cpg_to_geometric_data
from ..transformer.code_generator import CodeTransformer, RefactoringGenerator
from ..xai.explanation_module import CausalInferenceModule, ExplanationGenerator, ExplainabilityReport
from ..utils.helpers import ConfigManager, LoggerSetup, MetricsTracker


class XRefactorPipeline:
    """
    Main pipeline orchestrator for XRefactor framework
    """
    
    def __init__(self, config_path: str, device: str = "cuda"):
        """
        Initialize XRefactor pipeline
        
        Args:
            config_path: Path to configuration YAML file
            device: "cuda" or "cpu"
        """
        self.config = ConfigManager(config_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.metrics = MetricsTracker()
        
        logger.info(f"Initializing XRefactor Pipeline on {self.device}")
        
        # Initialize components
        self.cpg: Optional[CodePropertyGraph] = None
        self.node_encoder: Optional[DependencyHypergraphEncoder] = None
        self.gnn_model: Optional[HypergraphGNN] = None
        self.refactoring_predictor: Optional[RefactoringPredictor] = None
        self.transformer: Optional[CodeTransformer] = None
        self.xai_module: Optional[CausalInferenceModule] = None
        
        self.node_type_to_id: Dict[str, int] = self._create_node_type_mapping()
        self.node_refactoring_logits: Optional[torch.Tensor] = None
        self.node_confidence_scores: Optional[torch.Tensor] = None
        self.detected_smells: List[Dict[str, Any]] = []
        self.transformer_model_aliases = {
            "codebert": "microsoft/codebert-base",
            "graphcodebert": "microsoft/graphcodebert-base"
        }
    
    def _create_node_type_mapping(self) -> Dict[str, int]:
        """Create mapping from node types to IDs"""
        return {
            "class": 0,
            "method": 1,
            "function": 2,
            "field": 3,
            "variable": 4,
            "statement": 5,
            "call": 6,
            "definition": 7
        }
    
    def _resolve_transformer_model_name(self, model_type: str) -> str:
        """Resolve transformer model alias to a Hugging Face model identifier"""
        if not model_type:
            return "microsoft/codebert-base"
        normalized = model_type.strip().lower()
        return self.transformer_model_aliases.get(normalized, model_type)
    
    def _load_gnn_checkpoint_if_available(self) -> None:
        """
        Load self-supervised pretrained weights (see src/gnn/pretrain.py) into the
        node encoder and GNN if a checkpoint is configured. Without this, both
        modules stay randomly initialized, so we log which mode is active rather
        than silently pretending the embeddings are meaningful.
        """
        checkpoint_path = self.config.get("gnn.pretrained_checkpoint")
        if not checkpoint_path:
            logger.warning(
                "No gnn.pretrained_checkpoint configured - GNN/node encoder weights "
                "are randomly initialized. Structural embeddings will be shape-correct "
                "but not semantically meaningful. Run src/gnn/pretrain.py to train one."
            )
            return
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"gnn.pretrained_checkpoint '{checkpoint_path}' not found - using random init")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.node_encoder.load_state_dict(checkpoint["node_encoder_state"])
        self.gnn_model.load_state_dict(checkpoint["gnn_model_state"])
        logger.info(f"Loaded pretrained GNN weights from {checkpoint_path}")
    
    def _load_refactoring_predictor_checkpoint_if_available(self) -> None:
        """
        Load classifier/confidence weights trained on real mined refactorings
        (see refactoring_mining/train_refactoring_predictor.py) if configured.
        Without this, refactoring-type predictions have no supervision signal.
        """
        checkpoint_path = self.config.get("gnn.refactoring_predictor_checkpoint")
        if not checkpoint_path:
            logger.warning(
                "No gnn.refactoring_predictor_checkpoint configured - refactoring type/"
                "confidence predictions are untrained. See refactoring_mining/ for how "
                "to mine real labels and train this head."
            )
            return
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"gnn.refactoring_predictor_checkpoint '{checkpoint_path}' not found - using random init")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.refactoring_predictor.classifier.load_state_dict(checkpoint["classifier_state"])
        self.refactoring_predictor.confidence_predictor.load_state_dict(checkpoint["confidence_predictor_state"])
        logger.info(f"Loaded trained RefactoringPredictor weights from {checkpoint_path}")

    def _load_transformer_checkpoint_if_available(self) -> None:
        """
        Load decoder/fusion weights trained on real mined before/after code pairs
        (see src/transformer/train_transformer.py) if configured. Without this,
        the decoder is randomly initialized and generation is not meaningful.

        Note: that training script conditions the decoder on a per-refactoring-
        TYPE placeholder vector (no per-example structural embedding was used,
        for tractability - see its docstring), while this pipeline passes real
        per-node GNN embeddings here. The trained gnn_projection/fusion_layer
        still define a genuine learned mapping into the decoder's attention
        space, but expect a distribution shift versus what training saw.
        """
        checkpoint_path = self.config.get("transformer.pretrained_checkpoint")
        if not checkpoint_path:
            logger.warning(
                "No transformer.pretrained_checkpoint configured - transformer decoder weights "
                "are randomly initialized. Generated code will not be meaningful. Run "
                "src/transformer/train_transformer.py to train one."
            )
            return

        if not os.path.exists(checkpoint_path):
            logger.warning(f"transformer.pretrained_checkpoint '{checkpoint_path}' not found - using random init")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.transformer.decoder.load_state_dict(checkpoint["decoder_state"])
        self.transformer.fusion_layer.load_state_dict(checkpoint["fusion_layer_state"])
        self.transformer.gnn_projection.load_state_dict(checkpoint["gnn_projection_state"])
        logger.info(f"Loaded trained transformer weights from {checkpoint_path}")

    def _load_xai_checkpoint_if_available(self) -> None:
        """
        Load problem_detector/solution_evaluator weights trained on weak/proxy
        supervision from real mined refactorings (see
        src/xai/train_causal_module.py) if configured. Without this, both
        heads are randomly initialized, so problem/solution scores (and the
        confidence_score they feed into) are not meaningful - only the
        template-based explanation TEXT (which is keyed off refactoring_type,
        not these scores) stays reasonable either way.
        """
        checkpoint_path = self.config.get("xai.pretrained_checkpoint")
        if not checkpoint_path:
            logger.warning(
                "No xai.pretrained_checkpoint configured - problem_detector/solution_evaluator "
                "weights are randomly initialized. Evidence-card confidence scores are not "
                "meaningful. Run src/xai/train_causal_module.py to train one."
            )
            return

        if not os.path.exists(checkpoint_path):
            logger.warning(f"xai.pretrained_checkpoint '{checkpoint_path}' not found - using random init")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.xai_module.problem_detector.load_state_dict(checkpoint["problem_detector_state"])
        self.xai_module.solution_evaluator.load_state_dict(checkpoint["solution_evaluator_state"])
        logger.info(f"Loaded trained XAI weights from {checkpoint_path}")

    def stage_1_cpg_construction(self, data_directory: str) -> CodePropertyGraph:
        """
        Stage 1: Construct Code Property Graph
        
        Args:
            data_directory: Root directory containing source code
        
        Returns:
            Constructed CPG
        """
        logger.info("=" * 80)
        logger.info("STAGE 1: CODE PROPERTY GRAPH CONSTRUCTION")
        logger.info("=" * 80)
        
        language = self.config.get("cpg.language", "java")
        include_data_flow = self.config.get("cpg.include_data_flow", True)
        include_control_flow = self.config.get("cpg.include_control_flow", True)
        include_call_graph = self.config.get("cpg.include_call_graph", True)
        
        self.cpg = CodePropertyGraph(
            language=language,
            include_data_flow=include_data_flow,
            include_control_flow=include_control_flow,
            include_call_graph=include_call_graph
        )
        
        self.cpg.build_from_directory(data_directory)
        
        # Log statistics
        stats = self.cpg.get_statistics()
        logger.info(f"CPG Construction Complete:")
        logger.info(f"  - Nodes: {stats['total_nodes']}")
        logger.info(f"  - Edges: {stats['total_edges']}")
        logger.info(f"  - Files: {stats['total_files']}")
        logger.info(f"  - Node types: {stats['node_types']}")
        
        for metric_name, value in stats.items():
            if isinstance(value, (int, float)):
                self.metrics.record(f"cpg_{metric_name}", float(value))

        # Rule-based smell detection: a non-learned, auditable counterpart to
        # Stage 2's GNN-predicted refactoring_type (see src/cpg/smell_detector.py).
        self.detected_smells = SmellDetector(self.config).detect(self.cpg)
        smell_counts = Counter(s["smell_type"] for s in self.detected_smells)
        logger.info(f"Rule-based smell detection: {len(self.detected_smells)} findings - {dict(smell_counts)}")
        self.metrics.record("cpg_detected_smells_total", float(len(self.detected_smells)))

        return self.cpg
    
    def stage_2_gnn_reasoning(self, cpg: CodePropertyGraph) -> Tuple[Any, Any]:
        """
        Stage 2: Graph Neural Network Reasoning
        
        Args:
            cpg: Code Property Graph from stage 1
        
        Returns:
            Tuple of (GNN model, node embeddings)
        """
        logger.info("=" * 80)
        logger.info("STAGE 2: GRAPH NEURAL NETWORK REASONING")
        logger.info("=" * 80)
        
        # Create GNN model
        hidden_dims = self.config.get("gnn.hidden_dims", [256, 256])
        output_dim = self.config.get("gnn.output_dim", 128)
        model_type = self.config.get("gnn.model_type", "gat")
        num_heads = self.config.get("gnn.num_heads", 8)
        dropout = self.config.get("gnn.dropout", 0.1)
        num_layers = self.config.get("gnn.num_layers", 3)
        node_embedding_dim = self.config.get("gnn.node_embedding_dim", 64)
        
        # Learned node-type embeddings instead of a raw ordinal type ID, so the GNN
        # receives a meaningful categorical signal rather than an arbitrary scalar
        self.node_encoder = DependencyHypergraphEncoder(
            node_type_vocab_size=len(self.node_type_to_id),
            embedding_dim=node_embedding_dim
        ).to(self.device)
        
        self.gnn_model = HypergraphGNN(
            input_dim=node_embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            model_type=model_type,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers
        ).to(self.device)
        
        self._load_gnn_checkpoint_if_available()
        
        # Convert CPG to geometric data
        cpg_dict = cpg.to_dict()
        geometric_data = convert_cpg_to_geometric_data(cpg_dict, self.node_type_to_id, encoder=self.node_encoder)
        
        logger.info(f"GNN Model created: {model_type.upper()}")
        logger.info(f"  - Input dim: {node_embedding_dim}")
        logger.info(f"  - Hidden dims: {hidden_dims}")
        logger.info(f"  - Output dim: {output_dim}")
        
        # Refactoring predictor: learned per-node refactoring type + confidence,
        # replacing the previous arbitrary round-robin/fixed-confidence heuristic
        self.refactoring_predictor = RefactoringPredictor(
            self.gnn_model,
            output_dim=output_dim,
            num_refactoring_types=10
        ).to(self.device)
        
        self._load_refactoring_predictor_checkpoint_if_available()
        
        # Forward pass to get embeddings
        self.gnn_model.eval()
        self.refactoring_predictor.eval()
        with torch.no_grad():
            x = geometric_data.x.float().to(self.device)
            edge_index = geometric_data.edge_index.to(self.device)
            
            node_embeddings, graph_embedding = self.gnn_model(x, edge_index)
            refactoring_logits, confidence_scores = self.refactoring_predictor(x, edge_index)
            
            logger.info(f"GNN Inference Complete:")
            logger.info(f"  - Node embeddings shape: {node_embeddings.shape}")
            logger.info(f"  - Graph embedding shape: {graph_embedding.shape}")
        
        self.node_refactoring_logits = refactoring_logits
        self.node_confidence_scores = confidence_scores
        
        self.metrics.record("gnn_nodes", float(node_embeddings.shape[0]))
        self.metrics.record("gnn_embedding_dim", float(node_embeddings.shape[1]))
        
        return self.gnn_model, node_embeddings
    
    def stage_3_transformer_generation(self, 
                                       source_code_samples: List[Dict[str, Any]],
                                       node_embeddings: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Stage 3: Transformer-based Code Generation
        
        Args:
            source_code_samples: List of source code snippets with metadata
            node_embeddings: Node embeddings from GNN
        
        Returns:
            List of refactoring suggestions
        """
        logger.info("=" * 80)
        logger.info("STAGE 3: TRANSFORMER-BASED CODE GENERATION")
        logger.info("=" * 80)
        
        raw_model_type = self.config.get("transformer.model_type", "codebert")
        model_name = self._resolve_transformer_model_name(raw_model_type)
        hidden_size = self.config.get("transformer.hidden_size", 768)
        num_layers = self.config.get("transformer.num_layers", 6)
        num_heads = self.config.get("transformer.num_attention_heads", 12)
        max_length = self.config.get("transformer.max_length", 256)
        use_ast_engine = self.config.get("refactor.use_ast_engine", True)
        
        gnn_embedding_dim = node_embeddings.shape[1]
        self.transformer = CodeTransformer(
            model_name=model_name,
            gnn_embedding_dim=gnn_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_heads
        ).to(self.device)
        self._load_transformer_checkpoint_if_available()

        logger.info(f"Transformer model alias '{raw_model_type}' resolved to '{model_name}'")
        logger.info(f"Transformer Model created:")
        logger.info(f"  - Pre-trained model: {model_name}")
        logger.info(f"  - Hidden size: {hidden_size}")
        logger.info(f"  - Layers: {num_layers}")
        
        # Create generator
        generator = RefactoringGenerator(self.transformer, device=self.device)
        pattern_name_to_id = {name: type_id for type_id, name in generator.refactoring_types.items()}

        # Rule-based smell detector (Stage 1) vs. the GNN's learned prediction
        # (Stage 2): the highest-severity detected smell per node, if any.
        smells_by_node_id: Dict[str, Dict[str, Any]] = {}
        for smell in self.detected_smells:
            node_id = smell["node_id"]
            if node_id not in smells_by_node_id or smell["severity_score"] > smells_by_node_id[node_id]["severity_score"]:
                smells_by_node_id[node_id] = smell

        max_suggestions = self.config.get("refactor.max_suggestions", 5)
        refactoring_suggestions = []
        neural_backlog = []  # candidates the engine declined, used to fill remaining slots

        for idx, sample in enumerate(source_code_samples):
            if len(refactoring_suggestions) >= max_suggestions:
                break
            # Prefer the FULL declaration source. A single line (all
            # code_snippet holds) is not a refactorable unit, and the
            # transformer was trained on whole method bodies.
            source_code = (
                sample.get("source_text")
                or sample.get("code_snippet")
                or sample.get("snippet", "")
            )
            if not source_code:
                continue
            # Use the sample's true position in the CPG node list, not its position
            # in this filtered sample list, so it lines up with node_embeddings' rows
            node_index = sample.get("node_index", idx)
            embedding_index = node_index if node_index < node_embeddings.shape[0] else -1
            source_embedding = node_embeddings[embedding_index].unsqueeze(0)

            if self.node_refactoring_logits is not None and embedding_index < self.node_refactoring_logits.shape[0]:
                learned_type_id = int(torch.argmax(self.node_refactoring_logits[embedding_index]).item())
                confidence_score = float(self.node_confidence_scores[embedding_index].item())
            else:
                learned_type_id = idx % len(generator.refactoring_types)
                confidence_score = 0.75 + 0.05 * (idx % 5)
            learned_pattern = generator.refactoring_types.get(learned_type_id, "unknown")

            # The rule-based smell detector, when it has an opinion for this exact
            # node, drives the actual pattern used for generation - it is the more
            # directly auditable signal (traces to a concrete metric, not just a
            # classifier score). The learned prediction is still recorded so
            # agreement/disagreement is visible, not silently discarded.
            detected_smell = smells_by_node_id.get(sample.get("node_id"))
            if detected_smell is not None:
                detected_pattern = detected_smell["refactoring_pattern"]
                refactoring_type = pattern_name_to_id.get(detected_pattern, learned_type_id)
            else:
                detected_pattern = None
                refactoring_type = learned_type_id

            pattern_name = generator.refactoring_types.get(refactoring_type, "unknown")

            # HYBRID ROUTING. Try the deterministic engine first: when it applies
            # AND javac verifies the edit, the refactoring is correct by
            # construction rather than merely plausible. Otherwise fall back to
            # the neural generator, recording why the engine declined.
            engine_outcome = None
            predicted_pattern = pattern_name
            pattern_source = "smell" if detected_pattern else "learned"
            opportunity_refusals = None

            if use_ast_engine and sample.get("type") == "method" and sample.get("name"):
                if pattern_name in SUPPORTED_PATTERNS:
                    engine_outcome = attempt_refactoring(
                        pattern=pattern_name,
                        member_name=sample["name"],
                        file_path=sample.get("file"),
                        member_line=sample.get("line"),
                    )
                if engine_outcome is None or not engine_outcome.usable:
                    # The predicted pattern has no deterministic implementation or
                    # the engine refused it. Ask whether ANY verifiable refactoring
                    # applies here - a verified correct edit beats an unverified
                    # generated one. Recorded explicitly: the applied pattern then
                    # differs from the predicted one, and both are reported.
                    found, found_outcome, opportunity_refusals = find_applicable_pattern(
                        member_name=sample["name"],
                        file_path=sample.get("file"),
                        member_line=sample.get("line"),
                    )
                    if found is not None:
                        pattern_name = found
                        refactoring_type = pattern_name_to_id.get(found, refactoring_type)
                        engine_outcome = found_outcome
                        pattern_source = "ast_opportunity"

            if engine_outcome is not None and engine_outcome.usable:
                suggestion = {
                    "original_code": engine_outcome.result.original_member or source_code,
                    "refactored_code": engine_outcome.result.refactored_member
                                        if engine_outcome.result.refactored_member is not None
                                        else "",
                    "refactoring_type": pattern_name,
                    # Deterministic and compile-verified: this is not a model
                    # score, and S12's reporting excludes it from calibration.
                    "confidence": 1.0,
                    "raw_confidence": float(confidence_score),
                    "confidence_kind": "deterministic",
                    "type_id": refactoring_type,
                    "generation_path": "ast",
                    "ast_refusal_code": None,
                    "verification": engine_outcome.verification.to_dict(),
                    "edit": engine_outcome.result.metadata,
                    "structural_validation": validate_pattern(
                        pattern_name,
                        engine_outcome.result.original_member or source_code,
                        engine_outcome.result.refactored_member or "",
                    ),
                }
            else:
                # Defer neural generation: let the engine scan the whole ranked
                # candidate pool first, so verified refactorings are not crowded
                # out by unverified ones from earlier candidates. Generation is
                # also the expensive step, so doing it last avoids paying for
                # candidates we never emit.
                neural_backlog.append({
                    "sample": sample,
                    "source_code": source_code,
                    "source_embedding": source_embedding,
                    "confidence_score": confidence_score,
                    "refactoring_type": refactoring_type,
                    "pattern_name": pattern_name,
                    "predicted_pattern": predicted_pattern,
                    "pattern_source": pattern_source,
                    "embedding_index": embedding_index,
                    "learned_pattern": learned_pattern,
                    "detected_pattern": detected_pattern,
                    "engine_outcome": engine_outcome,
                    "opportunity_refusals": opportunity_refusals,
                })
                continue

            suggestion["source_file"] = sample.get("file")
            suggestion["node_type"] = sample.get("type")
            suggestion["embedding_index"] = embedding_index
            suggestion["learned_pattern"] = learned_pattern
            suggestion["detected_smell_pattern"] = detected_pattern
            suggestion["pattern_agreement"] = (detected_pattern is None) or (detected_pattern == learned_pattern)
            # Which signal chose the pattern, and what was originally predicted -
            # so an engine-opportunity override is never mistaken for the
            # classifier having been right.
            suggestion["pattern_source"] = pattern_source
            suggestion["predicted_pattern"] = predicted_pattern
            if opportunity_refusals:
                suggestion["ast_opportunity_refusals"] = [
                    {"pattern": p, "refusal": r} for p, r in opportunity_refusals
                ]
            refactoring_suggestions.append(suggestion)

        ast_count = len(refactoring_suggestions)

        # Fill any remaining slots with neural generation, so the neural path is
        # still represented (and measurable) rather than silently crowded out.
        for pending in neural_backlog:
            if len(refactoring_suggestions) >= max_suggestions:
                break
            engine_outcome = pending["engine_outcome"]
            suggestion = generator.suggest_refactoring(
                source_code=pending["source_code"],
                gnn_embeddings=pending["source_embedding"],
                confidence_score=pending["confidence_score"],
                refactoring_type=pending["refactoring_type"],
                max_length=max_length
            )
            suggestion["confidence_kind"] = "model"
            suggestion["generation_path"] = "neural"
            suggestion["ast_refusal_code"] = (
                engine_outcome.result.refusal_code if engine_outcome is not None
                else (None if pending["pattern_name"] in SUPPORTED_PATTERNS
                      else "PATTERN_NOT_SUPPORTED")
            )
            suggestion["verification"] = (
                engine_outcome.verification.to_dict()
                if engine_outcome is not None and engine_outcome.verification is not None
                else None
            )
            suggestion["edit"] = None
            suggestion["source_file"] = pending["sample"].get("file")
            suggestion["node_type"] = pending["sample"].get("type")
            suggestion["embedding_index"] = pending["embedding_index"]
            suggestion["learned_pattern"] = pending["learned_pattern"]
            suggestion["detected_smell_pattern"] = pending["detected_pattern"]
            suggestion["pattern_agreement"] = (
                (pending["detected_pattern"] is None)
                or (pending["detected_pattern"] == pending["learned_pattern"])
            )
            suggestion["pattern_source"] = pending["pattern_source"]
            suggestion["predicted_pattern"] = pending["predicted_pattern"]
            if pending["opportunity_refusals"]:
                suggestion["ast_opportunity_refusals"] = [
                    {"pattern": p, "refusal": r} for p, r in pending["opportunity_refusals"]
                ]
            refactoring_suggestions.append(suggestion)

        neural_count = len(refactoring_suggestions) - ast_count
        self.metrics.record("transformer_samples", float(len(refactoring_suggestions)))
        self.metrics.record("suggestions_ast_verified", float(ast_count))
        self.metrics.record("suggestions_neural", float(neural_count))
        logger.info(f"Generated {len(refactoring_suggestions)} refactoring suggestions "
                    f"({ast_count} deterministic+verified, {neural_count} neural) "
                    f"from {len(source_code_samples)} candidates")

        return refactoring_suggestions
    
    def stage_4_xai_explanation(self,
                                refactoring_suggestions: List[Dict[str, Any]],
                                node_embeddings: torch.Tensor) -> Dict[str, Any]:
        """
        Stage 4: Explainable AI (XAI) Module

        Args:
            refactoring_suggestions: Refactoring suggestions from stage 3
            node_embeddings: Node embeddings for explanation

        Returns:
            Report dict from ExplainabilityReport.generate_report(), with real
            evidence cards (previously this stage constructed the module but
            never called it - "num_evidence_cards" was just len(suggestions),
            not an actual count of generated cards).
        """
        logger.info("=" * 80)
        logger.info("STAGE 4: EXPLAINABLE AI MODULE")
        logger.info("=" * 80)

        # Create causal inference module
        feature_dim = node_embeddings.shape[1]
        self.xai_module = CausalInferenceModule(feature_dim=feature_dim).to(self.device)
        self._load_xai_checkpoint_if_available()
        self.xai_module.eval()

        # Create explanation generator + report generator
        explanation_gen = ExplanationGenerator(self.xai_module)
        report_gen = ExplainabilityReport(explanation_gen)

        logger.info(f"XAI Module created:")
        logger.info(f"  - Feature dimension: {feature_dim}")
        logger.info(f"  - Explanation method: causal_inference")
        logger.info(f"  - Number of suggestions: {len(refactoring_suggestions)}")

        evidence_cards = []
        with torch.no_grad():
            for suggestion in refactoring_suggestions:
                embedding_index = suggestion.get("embedding_index", -1)
                if embedding_index is None or embedding_index < 0 or embedding_index >= node_embeddings.shape[0]:
                    continue
                original_embedding = node_embeddings[embedding_index].unsqueeze(0).unsqueeze(0)  # [1, 1, feature_dim]
                # No embedding of the GENERATED code is available at inference time
                # (would require re-running CPG+GNN on the freshly generated snippet,
                # which stage 3's raw text output isn't set up to do) - the original
                # node's embedding stands in for both slots. solution_scores here
                # reflect "how good does this code look in isolation" rather than a
                # genuine before/after structural delta - a documented limitation,
                # not a hidden one. Training (src/xai/train_causal_module.py) does
                # use real distinct before/after embeddings, so the heads themselves
                # are meaningfully trained even though this specific input isn't.
                refactored_embedding = original_embedding

                problem_scores, solution_scores, attn_weights = self.xai_module(
                    original_embedding, refactored_embedding
                )

                evidence_card = explanation_gen.generate_evidence_card(
                    problem_scores=problem_scores.squeeze(0),
                    solution_scores=solution_scores.squeeze(0),
                    attention_weights=attn_weights.squeeze(0),
                    refactoring_type=suggestion.get("refactoring_type", "unknown"),
                    affected_files=[suggestion.get("source_file") or "unknown"],
                    cpg_nodes={}
                )
                evidence_cards.append(evidence_card)

        report = report_gen.generate_report(evidence_cards)
        self.metrics.record("xai_suggestions", float(len(refactoring_suggestions)))
        self.metrics.record("xai_evidence_cards", float(len(evidence_cards)))
        logger.info(f"Generated {len(evidence_cards)} real evidence cards")

        return report
    
    def run_pipeline(self, data_directory: str, output_directory: str = None) -> Dict[str, Any]:
        """
        Run the complete XRefactor pipeline
        
        Args:
            data_directory: Root directory containing source code
            output_directory: Directory to save outputs
        
        Returns:
            Pipeline results dictionary
        """
        if output_directory is None:
            output_directory = self.config.get("project.output_dir", "./outputs")
        
        os.makedirs(output_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "timestamp": timestamp,
            "stages": {}
        }
        
        try:
            # Stage 1: CPG Construction
            logger.info("\n" + "=" * 80)
            logger.info("STARTING XREFACTOR PIPELINE")
            logger.info("=" * 80 + "\n")
            
            cpg = self.stage_1_cpg_construction(data_directory)
            results["stages"]["cpg"] = {
                "status": "success",
                "statistics": cpg.get_statistics(),
                "graph": cpg.get_visualization_data()
            }
            
            # Stage 2: GNN Reasoning
            gnn_model, node_embeddings = self.stage_2_gnn_reasoning(cpg)
            results["stages"]["gnn"] = {
                "status": "success",
                "model_type": self.config.get("gnn.model_type", "gat"),
                "embedding_dim": node_embeddings.shape[1]
            }
            
            # Extract sample code for stage 3, keeping each node's true position so it
            # can be matched back to the corresponding row in node_embeddings.
            # Methods/classes are preferred over statement nodes: only they carry
            # `source_text` (a full refactorable declaration), and a lone
            # statement line is not something a refactoring can be applied to.
            cpg_dict = cpg.to_dict()
            smell_node_ids = {s["node_id"] for s in self.detected_smells}
            candidate_pool = self.config.get("refactor.candidate_pool", 40)

            candidates = []
            for node_index, (node_id, node) in enumerate(cpg_dict["nodes"].items()):
                if not (node.get("source_text") or node.get("code_snippet")):
                    continue
                # Nodes the smell detector actually flagged come first: examining
                # the first five nodes in file order was arbitrary and mostly
                # surfaced code with nothing wrong with it.
                flagged = 0 if node_id in smell_node_ids else 1
                rank = {"method": 0, "class": 1}.get(node.get("type"), 2)
                candidates.append((flagged, rank, node_index, node_id, node))
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))

            sample_code = [
                {
                    "code_snippet": node.get("code_snippet", ""),
                    "source_text": node.get("source_text", ""),
                    "file": node.get("file"),
                    "type": node.get("type"),
                    # name/line let the AST engine locate this exact declaration
                    # (the CPG keys methods by name alone, so overloads collide).
                    "name": node.get("name"),
                    "line": node.get("line"),
                    "node_index": node_index,
                    "node_id": node_id,
                }
                for _flagged, _rank, node_index, node_id, node in candidates[:candidate_pool]
            ]
            if not sample_code:
                sample_code = [{"code_snippet": f"// Sample {i}", "file": None, "type": "sample", "node_index": i} for i in range(min(5, len(cpg_dict["nodes"])))]
            
            # Stage 3: Transformer Generation
            refactoring_suggestions = self.stage_3_transformer_generation(sample_code, node_embeddings)
            results["stages"]["transformer"] = {
                "status": "success",
                "model": self._resolve_transformer_model_name(self.config.get("transformer.model_type", "codebert")),
                "refactoring_suggestions": refactoring_suggestions
            }
            
            # Stage 4: XAI Explanation
            xai_report = self.stage_4_xai_explanation(refactoring_suggestions, node_embeddings)
            results["stages"]["xai"] = {
                "status": "success",
                "explanation_method": "causal_inference",
                "num_evidence_cards": xai_report["num_suggestions"],
                "avg_confidence": xai_report["total_confidence"],
            }

            evidence_cards_file = os.path.join(output_directory, f"evidence_cards_{timestamp}.json")
            with open(evidence_cards_file, "w") as f:
                json.dump(xai_report, f, indent=2, default=str)
            logger.info(f"Evidence cards saved to {evidence_cards_file}")
            
            # Save results
            results["status"] = "success"
            results["metrics"] = self.metrics.get_all_statistics()
            
            self._save_pipeline_results(results, output_directory, timestamp)

            logger.info("\n" + "=" * 80)
            logger.info("XREFACTOR PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results

    def _resolve_transformer_model_name(self, model_type: str) -> str:
        """Resolve transformer model alias to actual Hugging Face model name"""
        if not model_type:
            return "microsoft/codebert-base"
        normalized = model_type.strip().lower()
        return self.transformer_model_aliases.get(normalized, model_type)
    
    def _save_pipeline_results(self, results: Dict[str, Any], 
                              output_dir: str, timestamp: str) -> None:
        """Save pipeline results to output directory"""
        output_file = os.path.join(output_dir, f"pipeline_results_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to {output_file}")
        
        # Also save metrics separately
        metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
        self.metrics.save_metrics(metrics_file)
