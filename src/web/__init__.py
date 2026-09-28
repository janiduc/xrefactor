"""
Web backend for XRefactor GUI
Provides Flask app and REST API endpoints
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from loguru import logger
import os


def create_app(config=None):
    """
    Create and configure Flask application
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured Flask app instance
    """
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), '../../templates'),
        static_folder=os.path.join(os.path.dirname(__file__), '../../static')
    )
    
    # Enable CORS for all routes
    CORS(app)
    
    # Configuration
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
    
    if config:
        app.config.update(config)
    
    # Register blueprints
    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "message": "XRefactor backend is running"})
    
    # Main UI route
    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"status": "error", "message": "Resource not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"status": "error", "message": "Internal server error"}), 500
    
    logger.info("Flask app created successfully")
    return app
