"""
XRefactor Main Entry Point
Run the complete explainable refactoring pipeline
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.helpers import ConfigManager, LoggerSetup
from src.core.pipeline import XRefactorPipeline
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="XRefactor: Explainable AI-driven Cross-file Code Refactoring Framework"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing source code repositories"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    LoggerSetup.setup(
        log_level=args.log_level,
        output_file=os.path.join(args.output, "xrefactor.log")
    )
    
    logger.info("XRefactor - Explainable AI-driven Cross-file Code Refactoring Framework")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Validate inputs
    if not os.path.isdir(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    if not os.path.isfile(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Initialize and run pipeline
        pipeline = XRefactorPipeline(
            config_path=args.config,
            device=args.device
        )
        
        results = pipeline.run_pipeline(
            data_directory=args.data_dir,
            output_directory=args.output
        )
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Status: {results['status']}")
        logger.info(f"Timestamp: {results['timestamp']}")
        
        if results['status'] == 'success':
            for stage, info in results['stages'].items():
                logger.info(f"  {stage}: {info.get('status', 'unknown')}")
            
            sys.exit(0)
        else:
            logger.error(f"Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
