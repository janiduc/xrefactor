"""
REST API endpoints for XRefactor pipeline
Maps HTTP requests to pipeline stages
"""

from flask import Blueprint, request, jsonify, send_file
from loguru import logger
from typing import Dict, Any
import os
import json
import time
from datetime import datetime
from pathlib import Path
import traceback

from ..core.pipeline import XRefactorPipeline
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


@api_bp.route('/results/<result_id>', methods=['GET'])
def get_result(result_id: str):
    """Get specific pipeline result"""
    try:
        result_dir = f"./outputs/{result_id}"
        result_file = os.path.join(result_dir, "results.json")
        
        if not os.path.exists(result_file):
            return jsonify({
                "status": "error",
                "message": f"Result not found: {result_id}"
            }), 404
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        return jsonify({
            "status": "success",
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error retrieving results: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


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
