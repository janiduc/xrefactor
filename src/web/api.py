"""
REST API endpoints for XRefactor pipeline
Maps HTTP requests to pipeline stages
"""

from flask import Blueprint, request, jsonify, send_file
from loguru import logger
from typing import Any, Dict, Optional
import os
import json
import time
from datetime import datetime
from pathlib import Path
import traceback

from ..core.pipeline import XRefactorPipeline
from ..gnn.refactoring_types import REFACTORING_TYPE_TO_ID
from ..refactor.engine import SUPPORTED_PATTERNS
from ..utils.helpers import ConfigManager

api_bp = Blueprint('api', __name__)

# Global pipeline instance (lazy-initialized)
_pipeline = None
_pipeline_lock = False
_pipeline_started_at = None
_pipeline_data_directory = None
_STALE_LOCK_SECONDS = 1800  # auto-release the lock if a run has been stuck this long


def get_pipeline(config_path: str = None, device: str = "cpu") -> XRefactorPipeline:
    """Get or create global pipeline instance"""
    global _pipeline
    
    if _pipeline is None:
        if config_path is None:
            config_path = "./configs/config.yaml"
        _pipeline = XRefactorPipeline(config_path, device)
        logger.info(f"Pipeline initialized on device: {device}")
    
    return _pipeline


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "API is running",
        "timestamp": datetime.now().isoformat()
    })


@api_bp.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config_path = request.args.get('config_path', './configs/config.yaml')
        config = ConfigManager(config_path)
        return jsonify({
            "status": "success",
            "config": config.config
        })
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@api_bp.route('/pipeline/stages', methods=['GET'])
def get_pipeline_stages():
    """Get description of all pipeline stages"""
    stages = {
        "stage_1": {
            "name": "Code Property Graph Construction",
            "description": "Parse source code and build dependency graphs",
            "inputs": ["source_code_directory"],
            "outputs": ["cpg_graph", "node_count", "edge_count"]
        },
        "stage_2": {
            "name": "Graph Neural Network Reasoning",
            "description": "Apply GNN (GAT/GCN/GraphSAGE) for structural reasoning",
            "inputs": ["cpg_graph"],
            "outputs": ["node_embeddings", "graph_embedding", "refactoring_predictions"]
        },
        "stage_3": {
            "name": "Transformer Code Generation",
            "description": "Generate refactored code using CodeBERT + Transformer decoder",
            "inputs": ["source_code", "gnn_embeddings"],
            "outputs": ["refactored_code_candidates"]
        },
        "stage_4": {
            "name": "Explainable AI Analysis",
            "description": "Generate explanations for refactoring recommendations",
            "inputs": ["refactored_code", "gnn_embeddings"],
            "outputs": ["explanations", "confidence_scores", "evidence_cards"]
        }
    }
    
    return jsonify({
        "status": "success",
        "stages": stages,
        "architecture": "https://github.com/xrefactor/architecture"
    })


@api_bp.route('/pipeline/status', methods=['GET'])
def pipeline_status():
    """Report whether a pipeline run is currently in progress (survives frontend page reloads)"""
    elapsed = (time.time() - _pipeline_started_at) if (_pipeline_lock and _pipeline_started_at) else 0
    return jsonify({
        "status": "success",
        "running": bool(_pipeline_lock),
        "elapsed_seconds": round(elapsed, 1),
        "data_directory": _pipeline_data_directory if _pipeline_lock else None
    })


@api_bp.route('/pipeline/run', methods=['POST'])
def run_pipeline():
    """
    Run the full XRefactor pipeline
    
    Request JSON:
    {
        "data_directory": "/path/to/source",
        "config_path": "./configs/config.yaml",
        "device": "cpu",
        "output_directory": "./outputs"
    }
    """
    global _pipeline_lock, _pipeline_started_at, _pipeline_data_directory
    
    if _pipeline_lock:
        elapsed = time.time() - _pipeline_started_at if _pipeline_started_at else 0
        if elapsed > _STALE_LOCK_SECONDS:
            logger.warning(f"Pipeline lock stale after {elapsed:.0f}s, auto-releasing")
            _pipeline_lock = False
        else:
            return jsonify({
                "status": "error",
                "message": f"Pipeline is already running ({elapsed:.0f}s elapsed on {_pipeline_data_directory}). Please wait for it to finish.",
                "elapsed_seconds": round(elapsed, 1)
            }), 429  # Too Many Requests
    
    try:
        _pipeline_lock = True
        _pipeline_started_at = time.time()
        
        data = request.get_json()
        data_directory = data.get('data_directory')
        _pipeline_data_directory = data_directory
        config_path = data.get('config_path', './configs/config.yaml')
        device = data.get('device', 'cpu')
        output_directory = data.get('output_directory', './outputs')
        
        if not data_directory:
            return jsonify({
                "status": "error",
                "message": "data_directory is required"
            }), 400
        
        if not os.path.isdir(data_directory):
            return jsonify({
                "status": "error",
                "message": f"Directory not found: {data_directory}"
            }), 400
        
        logger.info(f"Starting pipeline run: {data_directory}")
        
        # Get pipeline instance
        pipeline = get_pipeline(config_path, device)
        
        # Run pipeline
        results = pipeline.run_pipeline(data_directory, output_directory)
        
        logger.info(f"Pipeline completed successfully")
        
        return jsonify({
            "status": "success",
            "message": "Pipeline executed successfully",
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500
    
    finally:
        _pipeline_lock = False
        _pipeline_started_at = None
        _pipeline_data_directory = None


@api_bp.route('/pipeline/stage/<stage_name>', methods=['POST'])
def run_stage(stage_name: str):
    """
    Run a specific pipeline stage
    
    Supported stages: cpg_construction, gnn_reasoning, transformer_generation, xai_explanation
    """
    try:
        data = request.get_json()
        config_path = data.get('config_path', './configs/config.yaml')
        device = data.get('device', 'cpu')
        
        pipeline = get_pipeline(config_path, device)
        
        if stage_name == "cpg_construction":
            data_directory = data.get('data_directory')
            if not data_directory or not os.path.isdir(data_directory):
                return jsonify({"status": "error", "message": "Invalid data_directory"}), 400
            
            cpg = pipeline.stage_1_cpg_construction(data_directory)
            stats = cpg.get_statistics()
            
            return jsonify({
                "status": "success",
                "stage": "cpg_construction",
                "statistics": stats,
                "graph": cpg.get_visualization_data()
            })
        
        elif stage_name == "gnn_reasoning":
            if pipeline.cpg is None:
                return jsonify({"status": "error", "message": "CPG not constructed. Run cpg_construction first."}), 400
            
            gnn_model, embeddings = pipeline.stage_2_gnn_reasoning(pipeline.cpg)
            
            return jsonify({
                "status": "success",
                "stage": "gnn_reasoning",
                "gnn_model": str(gnn_model),
                "embeddings_shape": str(embeddings.shape)
            })
        
        else:
            return jsonify({
                "status": "error",
                "message": f"Unknown stage: {stage_name}"
            }), 400
    
    except Exception as e:
        logger.error(f"Stage execution error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# --------------------------------------------------------------------------- #
# Saved run artifacts
#
# The pipeline writes three sibling files per run, all keyed by one timestamp:
#     outputs/pipeline_results_<ts>.json
#     outputs/metrics_<ts>.json
#     outputs/evidence_cards_<ts>.json
# Nothing used to serve them. The previous /results/<id> route looked for
# `outputs/<id>/results.json`, a layout the pipeline has never produced, so it
# returned 404 for every input and ~30 saved runs were unreachable - the only
# way to see anything was to re-run a multi-minute pipeline.
# --------------------------------------------------------------------------- #

OUTPUTS_DIR = "./outputs"
_RESULT_PREFIX = "pipeline_results_"


def _outputs_dir() -> Path:
    return Path(OUTPUTS_DIR)


def _result_path(run_id: str) -> Path:
    return _outputs_dir() / f"{_RESULT_PREFIX}{run_id}.json"


def _known_run_ids() -> list:
    """Run ids (timestamps) on disk, newest first."""
    paths = _outputs_dir().glob(f"{_RESULT_PREFIX}*.json")
    return sorted((p.stem[len(_RESULT_PREFIX):] for p in paths), reverse=True)


def _resolve_run_id(run_id: str) -> Optional[str]:
    """Accept a literal timestamp or the alias 'latest'.

    Rejects anything that is not a known id rather than joining user input into
    a path, so a traversal attempt cannot reach outside outputs/.
    """
    known = _known_run_ids()
    if run_id == "latest":
        return known[0] if known else None
    return run_id if run_id in known else None


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_summary(run_id: str) -> dict:
    """Projection of one saved run, for the run picker.

    Reads the whole file but returns only summary fields: the newest result is
    ~1MB (mostly `original_code`/`refactored_code` strings), so the full payload
    must not be multiplied across every row of a listing.
    """
    results = _load_json(_result_path(run_id))
    cpg_stats = (results.get("stages", {}).get("cpg", {}) or {}).get("statistics", {}) or {}
    suggestions = (results.get("stages", {}).get("transformer", {}) or {}).get("refactoring_suggestions", []) or []
    metrics = results.get("metrics", {}) or {}

    def metric(name: str):
        entry = metrics.get(name)
        return entry.get("mean") if isinstance(entry, dict) else None

    return {
        "id": run_id,
        # `data_directory`/`repo_label` are absent from runs saved before the
        # pipeline started recording them - reported as null rather than guessed.
        "data_directory": results.get("data_directory"),
        "repo_label": results.get("repo_label"),
        "run_status": results.get("status"),
        "total_nodes": cpg_stats.get("total_nodes"),
        "total_edges": cpg_stats.get("total_edges"),
        "total_files": cpg_stats.get("total_files"),
        "suggestion_count": len(suggestions),
        "ast_count": sum(1 for s in suggestions if s.get("generation_path") == "ast"),
        "neural_count": sum(1 for s in suggestions if s.get("generation_path") == "neural"),
        "smell_total": metric("cpg_detected_smells_total"),
        "has_evidence": (_outputs_dir() / f"evidence_cards_{run_id}.json").exists(),
        "has_metrics": (_outputs_dir() / f"metrics_{run_id}.json").exists(),
    }


@api_bp.route('/runs', methods=['GET'])
def list_runs():
    """List saved pipeline runs, newest first."""
    try:
        runs = []
        for run_id in _known_run_ids():
            try:
                runs.append(_run_summary(run_id))
            except Exception as e:
                # One corrupt or truncated artifact must not blank the picker.
                logger.warning(f"Skipping unreadable run {run_id}: {e}")
        return jsonify({"status": "success", "count": len(runs), "runs": runs})
    except Exception as e:
        logger.error(f"Error listing runs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route('/runs/<run_id>', methods=['GET'])
def get_run(run_id: str):
    """Full saved pipeline-results payload for one run ('latest' accepted)."""
    resolved = _resolve_run_id(run_id)
    if resolved is None:
        return jsonify({"status": "error", "message": f"Run not found: {run_id}"}), 404
    try:
        return jsonify({"status": "success", "id": resolved,
                        "results": _load_json(_result_path(resolved))})
    except Exception as e:
        logger.error(f"Error reading run {resolved}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route('/runs/<run_id>/evidence', methods=['GET'])
def get_run_evidence(run_id: str):
    """Stage-4 evidence-card report for one run.

    Stage 4 builds a full report (cards, explanations, risk buckets, impact
    analysis) but `results["stages"]["xai"]` keeps only four scalars, so the
    report lived on disk and was reachable over no HTTP route at all - the
    reason the 'explainable' stage rendered nothing in the UI.
    """
    resolved = _resolve_run_id(run_id)
    if resolved is None:
        return jsonify({"status": "error", "message": f"Run not found: {run_id}"}), 404
    evidence_path = _outputs_dir() / f"evidence_cards_{resolved}.json"
    if not evidence_path.exists():
        # Runs predating the evidence-cards file legitimately have none.
        return jsonify({"status": "error", "id": resolved,
                        "message": f"No evidence cards saved for run {resolved}"}), 404
    try:
        return jsonify({"status": "success", "id": resolved,
                        "report": _load_json(evidence_path)})
    except Exception as e:
        logger.error(f"Error reading evidence for {resolved}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route('/results/<result_id>', methods=['GET'])
def get_result(result_id: str):
    """Deprecated alias for /runs/<id>, kept so existing links keep working."""
    return get_run(result_id)


# --------------------------------------------------------------------------- #
# Offline evaluation benchmarks
#
# These describe the MODELS, not any one run, and they come from a different
# corpus and date than the newest pipeline run - so every payload below is
# wrapped with its own provenance (source file, mtime, repo, sample size).
# Presenting them as part of a run would be a real methodological error, and
# the provenance block is what lets the UI prevent it.
# --------------------------------------------------------------------------- #

_MODELS_DIR = "./models"
_MINING_REPORT = "./refactoring_mining/mining_report.json"


def _provenance(path: Path, **extra) -> dict:
    stat = path.stat()
    return {
        "source_file": str(path).replace("\\", "/"),
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "size_bytes": stat.st_size,
        **extra,
    }


def _newest_evaluation_report() -> Optional[Path]:
    paths = sorted(_outputs_dir().glob("evaluation_report_*.json"), reverse=True)
    return paths[0] if paths else None


@api_bp.route('/benchmarks', methods=['GET'])
def get_benchmarks():
    """Offline evaluation, model metrics, mining dataset and the taxonomy.

    One aggregated response so the Evaluation view needs a single fetch.

    `taxonomy` is derived from the SAME constants the pipeline uses -
    REFACTORING_TYPE_TO_ID and the refactoring engine's SUPPORTED_PATTERNS -
    rather than duplicated as a literal here, so the "has a deterministic
    implementation" flag cannot drift when the engine gains a pattern.

    Deliberately NOT served: outputs/baseline_prefix_*.json. It is byte-identical
    to the evaluation report and referenced by no code; exposing it would invite a
    fabricated "vs. baseline" comparison. A real comparator is derivable from the
    class supports (majority-class accuracy) and is computed client-side.
    """
    payload: Dict[str, Any] = {
        "status": "success",
        "taxonomy": {
            "order": [name for name, _ in sorted(REFACTORING_TYPE_TO_ID.items(), key=lambda kv: kv[1])],
            "ids": dict(REFACTORING_TYPE_TO_ID),
            "ast_supported": sorted(SUPPORTED_PATTERNS),
        },
        "eval": None,
        "models": {},
        "mining": None,
        "available_reports": [p.stem.replace("evaluation_report_", "")
                              for p in sorted(_outputs_dir().glob("evaluation_report_*.json"), reverse=True)],
    }

    report_path = _newest_evaluation_report()
    if report_path is not None:
        try:
            report = _load_json(report_path)
            trace = report.get("end_to_end_trace", {}) or {}
            generation = report.get("stage_3_transformer", {}) or {}
            payload["eval"] = {
                "provenance": _provenance(
                    report_path,
                    timestamp=report.get("timestamp"),
                    # The repo the end-to-end trace ran on; the UI compares this
                    # against the loaded run's repo_label to flag a mismatch.
                    repo=trace.get("repo"),
                    sample_size=generation.get("eval_sample_size"),
                    test_size=(report.get("stage_2_classifier", {}) or {}).get("test_size"),
                ),
                "classifier": report.get("stage_2_classifier"),
                "generation": generation,
                "xai": report.get("stage_4_xai"),
                "trace": trace,
            }
        except Exception as e:
            logger.warning(f"Could not read evaluation report {report_path}: {e}")

    for key, filename in (("predictor", "refactoring_predictor_trained.metrics.json"),
                          ("transformer", "transformer_trained.metrics.json")):
        path = Path(_MODELS_DIR) / filename
        if not path.exists():
            continue
        try:
            payload["models"][key] = {"provenance": _provenance(path), **_load_json(path)}
        except Exception as e:
            logger.warning(f"Could not read model metrics {path}: {e}")

    mining_path = Path(_MINING_REPORT)
    if mining_path.exists():
        try:
            payload["mining"] = {"provenance": _provenance(mining_path), **_load_json(mining_path)}
        except Exception as e:
            logger.warning(f"Could not read mining report {mining_path}: {e}")

    return jsonify(payload)


@api_bp.route('/supported-languages', methods=['GET'])
def get_supported_languages():
    """Get supported programming languages"""
    languages = {
        "java": {"label": "Java", "extensions": [".java"], "supported": True},
        "python": {"label": "Python", "extensions": [".py"], "supported": True},
        "csharp": {"label": "C#", "extensions": [".cs"], "supported": False},
        "cpp": {"label": "C++", "extensions": [".cpp", ".h"], "supported": False},
        "javascript": {"label": "JavaScript", "extensions": [".js", ".ts"], "supported": False}
    }
    
    return jsonify({
        "status": "success",
        "languages": languages
    })


@api_bp.route('/gnn-models', methods=['GET'])
def get_gnn_models():
    """Get available GNN model types"""
    models = {
        "gat": {
            "name": "Graph Attention Networks",
            "description": "Multi-head attention mechanism over graph structure",
            "pros": ["Flexible", "High accuracy"],
            "cons": ["Memory intensive"]
        },
        "gcn": {
            "name": "Graph Convolutional Networks",
            "description": "Simple spectral convolution over graph neighborhoods",
            "pros": ["Fast", "Memory efficient"],
            "cons": ["Less expressive"]
        },
        "graphsage": {
            "name": "GraphSAGE",
            "description": "Scalable inductive learning via neighbor sampling",
            "pros": ["Scalable", "Inductive"],
            "cons": ["Slower training"]
        }
    }
    
    return jsonify({
        "status": "success",
        "models": models
    })


@api_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500
