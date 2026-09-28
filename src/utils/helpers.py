"""
Utility modules for data processing, configuration, and helpers
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger
import logging


class ConfigManager:
    """Manages XRefactor configuration"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"Configuration loaded from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def save(self, output_path: str) -> None:
        """Save configuration to file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f)
        logger.info(f"Configuration saved to {output_path}")


class LoggerSetup:
    """Configure logging for XRefactor"""
    
    @staticmethod
    def setup(log_level: str = "INFO", output_file: str = None) -> None:
        """Setup logging configuration"""
        logger.remove()  # Remove default handler
        
        # Console logging
        logger.add(
            lambda msg: print(msg, end=''),
            level=log_level,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
        
        # File logging
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            logger.add(
                output_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
            )


class DatasetAnalyzer:
    """Analyze dataset repositories"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.statistics = {}
    
    def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """Analyze a single repository"""
        stats = {
            "path": repo_path,
            "java_files": 0,
            "python_files": 0,
            "total_lines": 0,
            "total_classes": 0,
            "total_methods": 0,
            "avg_method_length": 0,
            "avg_class_size": 0
        }
        
        # Count files
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.java'):
                    stats['java_files'] += 1
                elif file.endswith('.py'):
                    stats['python_files'] += 1
        
        return stats
    
    def analyze_all(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all repositories in dataset"""
        results = {}
        
        for repo_dir in os.listdir(self.dataset_path):
            repo_path = os.path.join(self.dataset_path, repo_dir)
            if os.path.isdir(repo_path):
                logger.info(f"Analyzing {repo_dir}...")
                results[repo_dir] = self.analyze_repository(repo_path)
        
        return results


class MetricsTracker:
    """Track metrics during execution"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if metric_name not in self.metrics:
            return {}
        
        values = self.metrics[metric_name]
        import statistics
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        return {
            metric: self.get_statistics(metric)
            for metric in self.metrics.keys()
        }
    
    def save_metrics(self, output_path: str) -> None:
        """Save metrics to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.get_all_statistics(), f, indent=2)
        logger.info(f"Metrics saved to {output_path}")
