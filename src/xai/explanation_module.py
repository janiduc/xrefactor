"""
Stage 4: Explainable AI (XAI) Module
Provides decision rationales through dual-view causal inference and evidence cards.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger
import json


@dataclass
class EvidenceCard:
    """Represents an evidence card linking changes to problematic relationships"""
    evidence_id: str
    problem_type: str  # "code_smell", "architectural_violation", "duplicate_code", etc.
    affected_entities: List[str]  # File paths or node IDs
    explanation: str
    supporting_metrics: Dict[str, float]
    confidence_score: float
    recommendation: str


class CausalInferenceModule(torch.nn.Module):
    """
    Performs dual-view causal inference to explain refactoring recommendations
    """
    
    def __init__(self, feature_dim: int = 768, hidden_dim: int = 256):
        super(CausalInferenceModule, self).__init__()
        
        # View 1: Problematic pattern recognition
        self.problem_detector = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),  # Problem score
            torch.nn.Sigmoid()
        )
        
        # View 2: Solution quality prediction
        self.solution_evaluator = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),  # Solution quality
            torch.nn.Sigmoid()
        )
        
        # Attention for highlighting problematic code
        self.attention_layer = torch.nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        logger.info("Initialized CausalInferenceModule")
    
    def forward(self, 
                original_embeddings: torch.Tensor,
                refactored_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform causal analysis
        
        Args:
            original_embeddings: [batch_size, seq_len, feature_dim]
            refactored_embeddings: [batch_size, seq_len, feature_dim]
        
        Returns:
            problem_scores: [batch_size, seq_len]
            solution_scores: [batch_size, seq_len]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        # Detect problems in original code
        problem_scores = self.problem_detector(original_embeddings)  # [batch_size, seq_len, 1]
        problem_scores = problem_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Evaluate solution quality
        combined = torch.cat([original_embeddings, refactored_embeddings], dim=-1)
        solution_scores = self.solution_evaluator(combined)  # [batch_size, seq_len, 1]
        solution_scores = solution_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Attention to highlight relationships
        attn_output, attn_weights = self.attention_layer(
            query=original_embeddings,
            key=refactored_embeddings,
            value=refactored_embeddings
        )
        
        return problem_scores, solution_scores, attn_weights


class ExplanationGenerator:
    """
    Generates natural language explanations for refactoring recommendations
    """
    
    def __init__(self, causal_module: CausalInferenceModule, max_explanation_length: int = 200):
        self.causal_module = causal_module
        self.max_explanation_length = max_explanation_length
        
        # Problem templates
        self.problem_templates = {
            "code_duplication": "Code duplication detected in {entities}. Multiple identical or near-identical code blocks increase maintenance burden.",
            "long_method": "Method {entities} exceeds recommended length. Long methods are hard to understand, test, and maintain.",
            "feature_envy": "Class {entities} accesses too many members of another class, violating encapsulation principles.",
            "tight_coupling": "Tight coupling detected between {entities}. High interdependencies limit modularity and reusability.",
            "dead_code": "Unreachable or unused code found in {entities}. Dead code increases cognitive load during maintenance.",
            "low_cohesion": "Low cohesion in {entities}. Methods and fields should be closely related in purpose.",
            "naming_issue": "Poor naming convention in {entities}. Clear, expressive names improve code readability.",
            "architectural_violation": "Architectural violation: {entities}. Dependencies violate intended layer separation.",
        }
        
        # Solution templates
        self.solution_templates = {
            "extract_method": "Extract {entities} into a new method to improve readability and reusability.",
            "move_class": "Move {entities} to appropriate module to reduce coupling and improve organization.",
            "rename_variable": "Rename {entities} to better express intent and improve code clarity.",
            "consolidate_duplicate_code": "Consolidate duplicate code in {entities} into a single reusable component.",
            "remove_dead_code": "Remove unused code in {entities} to reduce maintenance overhead.",
            "simplify_condition": "Simplify complex conditions in {entities} to improve readability.",
            "split_class": "Split {entities} into multiple focused classes following single responsibility principle.",
            "extract_interface": "Extract interface from {entities} to reduce coupling and improve abstraction.",
            "reduce_coupling": "Reduce coupling between {entities} through dependency inversion or composition.",
            "improve_naming": "Improve naming in {entities} to express intent more clearly.",
        }
    
    def generate_evidence_card(self,
                              problem_scores: torch.Tensor,
                              solution_scores: torch.Tensor,
                              attention_weights: torch.Tensor,
                              refactoring_type: str,
                              affected_files: List[str],
                              cpg_nodes: Dict[str, Any]) -> EvidenceCard:
        """
        Generate evidence card for a refactoring recommendation
        
        Args:
            problem_scores: Problem severity scores [seq_len]
            solution_scores: Solution quality scores [seq_len]
            attention_weights: Attention weights for key relationships
            refactoring_type: Type of refactoring
            affected_files: Files affected by refactoring
            cpg_nodes: CPG nodes involved
        
        Returns:
            EvidenceCard with explanation and supporting evidence
        """
        # Identify problem type from scores
        problem_type = self._infer_problem_type(problem_scores, refactoring_type)
        
        # Extract affected entities
        affected_entities = self._extract_affected_entities(
            attention_weights, cpg_nodes, affected_files
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            problem_type=problem_type,
            refactoring_type=refactoring_type,
            affected_entities=affected_entities
        )
        
        # Calculate supporting metrics
        metrics = {
            "avg_problem_score": float(problem_scores.mean().item()),
            "avg_solution_score": float(solution_scores.mean().item()),
            "improvement_delta": float((solution_scores - problem_scores).mean().item()),
            "attention_concentration": float(attention_weights.max().item()),
        }
        
        # Calculate confidence
        confidence = (metrics["improvement_delta"] + metrics["avg_solution_score"]) / 2.0
        confidence = min(1.0, max(0.0, confidence))
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            refactoring_type=refactoring_type,
            affected_entities=affected_entities,
            confidence=confidence
        )
        
        return EvidenceCard(
            evidence_id=f"ev_{refactoring_type}_{hash(tuple(affected_files)) % 10000}",
            problem_type=problem_type,
            affected_entities=affected_entities,
            explanation=explanation,
            supporting_metrics=metrics,
            confidence_score=confidence,
            recommendation=recommendation
        )
    
    def _infer_problem_type(self, problem_scores: torch.Tensor, refactoring_type: str) -> str:
        """Infer the problem type from scores and refactoring type"""
        problem_level = problem_scores.mean().item()
        
        if refactoring_type == "consolidate_duplicate_code" or problem_level > 0.7:
            return "code_duplication"
        elif refactoring_type == "extract_method":
            return "long_method"
        elif refactoring_type == "reduce_coupling":
            return "tight_coupling"
        elif refactoring_type == "remove_dead_code":
            return "dead_code"
        elif refactoring_type == "split_class":
            return "low_cohesion"
        elif refactoring_type == "move_class":
            return "architectural_violation"
        else:
            return "code_smell"
    
    def _extract_affected_entities(self, 
                                   attention_weights: torch.Tensor,
                                   cpg_nodes: Dict[str, Any],
                                   affected_files: List[str]) -> List[str]:
        """Extract affected entities based on attention weights"""
        # Simple extraction: top-k entities by attention
        if attention_weights.dim() > 1:
            attention_scores = attention_weights.mean(dim=0)
        else:
            attention_scores = attention_weights
        
        top_k = min(3, len(attention_scores))
        top_indices = torch.topk(attention_scores, top_k).indices.tolist()
        
        affected = affected_files[:top_k]
        return affected
    
    def _generate_explanation(self, problem_type: str, refactoring_type: str, 
                             affected_entities: List[str]) -> str:
        """Generate natural language explanation"""
        entities_str = ", ".join(affected_entities[:2])  # Limit to 2 for readability
        
        problem_desc = self.problem_templates.get(
            problem_type,
            f"Code quality issue detected in {entities_str}."
        ).format(entities=entities_str)
        
        solution_desc = self.solution_templates.get(
            refactoring_type,
            f"Apply {refactoring_type} refactoring to improve code quality."
        ).format(entities=entities_str)
        
        explanation = f"{problem_desc} {solution_desc}"
        
        if len(explanation) > self.max_explanation_length:
            explanation = explanation[:self.max_explanation_length] + "..."
        
        return explanation
    
    def _generate_recommendation(self, refactoring_type: str, 
                                affected_entities: List[str],
                                confidence: float) -> str:
        """Generate specific recommendation"""
        entities_str = ", ".join(affected_entities[:2])
        
        if confidence > 0.9:
            recommendation = f"Highly recommended: Apply {refactoring_type} to {entities_str}"
        elif confidence > 0.7:
            recommendation = f"Recommended: Consider applying {refactoring_type} to {entities_str}"
        else:
            recommendation = f"Optional: You might consider {refactoring_type} on {entities_str}"
        
        return recommendation


class ExplainabilityReport:
    """
    Generates comprehensive explainability reports for refactoring suggestions
    """
    
    def __init__(self, explanation_generator: ExplanationGenerator):
        self.generator = explanation_generator
    
    def generate_report(self, evidence_cards: List[EvidenceCard]) -> Dict[str, Any]:
        """
        Generate comprehensive explainability report
        
        Returns:
            Report with evidence, explanations, and recommendations
        """
        report = {
            "num_suggestions": len(evidence_cards),
            "total_confidence": sum(card.confidence_score for card in evidence_cards) / max(1, len(evidence_cards)),
            "evidence_cards": [asdict(card) for card in evidence_cards],
            "summary": self._generate_summary(evidence_cards),
            "risk_assessment": self._assess_risks(evidence_cards),
            "impact_analysis": self._analyze_impact(evidence_cards)
        }
        
        return report
    
    def _generate_summary(self, evidence_cards: List[EvidenceCard]) -> str:
        """Generate summary of all evidence"""
        if not evidence_cards:
            return "No refactoring suggestions generated."
        
        problem_types = set(card.problem_type for card in evidence_cards)
        refactoring_types = set(
            card.recommendation.split()[2] for card in evidence_cards if len(card.recommendation.split()) > 2
        )
        
        summary = f"Found {len(evidence_cards)} refactoring opportunities:\n"
        summary += f"- Problem types: {', '.join(problem_types)}\n"
        summary += f"- Recommended refactorings: {', '.join(refactoring_types)}"
        
        return summary
    
    def _assess_risks(self, evidence_cards: List[EvidenceCard]) -> Dict[str, Any]:
        """Assess risks of applying refactorings"""
        return {
            "high_risk": sum(1 for card in evidence_cards if card.confidence_score < 0.5),
            "medium_risk": sum(1 for card in evidence_cards if 0.5 <= card.confidence_score < 0.8),
            "low_risk": sum(1 for card in evidence_cards if card.confidence_score >= 0.8),
        }
    
    def _analyze_impact(self, evidence_cards: List[EvidenceCard]) -> Dict[str, Any]:
        """Analyze potential impact of refactorings"""
        affected_entities = set()
        for card in evidence_cards:
            affected_entities.update(card.affected_entities)
        
        return {
            "files_affected": len(affected_entities),
            "estimated_complexity": "low" if len(affected_entities) < 3 else "high",
            "total_improvements": sum(
                card.supporting_metrics.get("improvement_delta", 0) 
                for card in evidence_cards
            )
        }
    
    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Save report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")
    
    def format_report(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable string"""
        formatted = "=" * 80 + "\n"
        formatted += "XREFACTOR EXPLAINABILITY REPORT\n"
        formatted += "=" * 80 + "\n\n"
        
        formatted += f"Summary: {report['summary']}\n\n"
        
        formatted += "Evidence Cards:\n"
        formatted += "-" * 80 + "\n"
        for i, card in enumerate(report['evidence_cards'], 1):
            formatted += f"\n{i}. {card['problem_type'].upper()}\n"
            formatted += f"   Confidence: {card['confidence_score']:.2%}\n"
            formatted += f"   Explanation: {card['explanation']}\n"
            formatted += f"   Recommendation: {card['recommendation']}\n"
        
        formatted += "\n" + "=" * 80 + "\n"
        formatted += "Risk Assessment:\n"
        for risk_level, count in report['risk_assessment'].items():
            formatted += f"  {risk_level}: {count}\n"
        
        formatted += "\nImpact Analysis:\n"
        for impact_key, value in report['impact_analysis'].items():
            formatted += f"  {impact_key}: {value}\n"
        
        return formatted
