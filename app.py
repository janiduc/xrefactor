"""
XRefactor Web Application Entry Point
Runs the Flask backend server with GUI frontend
"""

import os
import sys
from argparse import ArgumentParser
from loguru import logger
from src.web import create_app
from src.utils.helpers import LoggerSetup


def main():
    parser = ArgumentParser(description="XRefactor Web Application")
    parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-level', default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--config', default='./configs/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    LoggerSetup.setup(log_level=args.log_level)
    
    logger.info("=" * 80)
    logger.info("XREFACTOR - EXPLAINABLE AI-DRIVEN CROSS-FILE CODE REFACTORING FRAMEWORK")
    logger.info("=" * 80)
    logger.info(f"Web Application Starting...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Debug Mode: {args.debug}")
    logger.info(f"Config: {args.config}")
    logger.info("")
    logger.info("Open your browser and navigate to:")
    logger.info(f"http://{args.host}:{args.port}")
    logger.info("")
    
    # Create Flask app
    app = create_app({
        'CONFIG_PATH': args.config,
        'DEBUG': args.debug
    })
    
    # Run server
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=args.debug
        )
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("Server shutdown requested by user")
        logger.info("=" * 80)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
