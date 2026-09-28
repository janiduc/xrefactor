"""
Web API Tests
Tests for Flask endpoints and API functionality
"""

import pytest
import json
import os
from pathlib import Path
from src.web import create_app


@pytest.fixture
def app():
    """Create app for testing"""
    app = create_app({'TESTING': True})
    return app


@pytest.fixture
def client(app):
    """Flask test client"""
    return app.test_client()


class TestAPIHealth:
    """Test health check endpoints"""
    
    def test_health_check(self, client):
        """Test /api/health endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
    
    def test_main_page_loads(self, client):
        """Test main page loads"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'XRefactor' in response.data


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_get_pipeline_stages(self, client):
        """Test /api/pipeline/stages endpoint"""
        response = client.get('/api/pipeline/stages')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'stages' in data
        assert 'stage_1' in data['stages']
        assert 'stage_2' in data['stages']
        assert 'stage_3' in data['stages']
        assert 'stage_4' in data['stages']
    
    def test_get_supported_languages(self, client):
        """Test /api/supported-languages endpoint"""
        response = client.get('/api/supported-languages')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'languages' in data
        assert 'java' in data['languages']
    
    def test_get_gnn_models(self, client):
        """Test /api/gnn-models endpoint"""
        response = client.get('/api/gnn-models')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'models' in data
        assert 'gat' in data['models']
        assert 'gcn' in data['models']
        assert 'graphsage' in data['models']
    
    def test_get_config(self, client):
        """Test /api/config endpoint"""
        response = client.get('/api/config')
        # Should succeed even if config file doesn't exist (empty dict)
        assert response.status_code == 200 or response.status_code == 400
    
    def test_pipeline_run_missing_directory(self, client):
        """Test pipeline run without data directory"""
        response = client.post('/api/pipeline/run', 
            json={
                'data_directory': None,
                'device': 'cpu'
            })
        assert response.status_code == 400
        data = response.get_json()
        assert data['status'] == 'error'
    
    def test_pipeline_run_invalid_directory(self, client):
        """Test pipeline run with invalid directory"""
        response = client.post('/api/pipeline/run',
            json={
                'data_directory': '/nonexistent/path',
                'device': 'cpu'
            })
        assert response.status_code == 400
        data = response.get_json()
        assert data['status'] == 'error'


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_not_found(self, client):
        """Test 404 error"""
        response = client.get('/api/nonexistent')
        assert response.status_code == 404
        data = response.get_json()
        assert data['status'] == 'error'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
