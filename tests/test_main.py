"""
Tests for FastAPI application endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch

from app.main import app
from app.schemas import MarketingFeature


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "choices": [
            {
                "message": {
                    "content": '''[
                        {
                            "id": "f1",
                            "name": "Portable Design",
                            "type": "benefit",
                            "importance_rank": 1,
                            "confidence": 0.95,
                            "example_phrase": "Fits in your hand"
                        }
                    ]''',
                    "role": "assistant"
                }
            }
        ]
    }


@pytest.fixture
def mock_storyline_response():
    """Mock storyline LLM response for testing."""
    return {
        "choices": [
            {
                "message": {
                    "content": '''{
                        "headline": "Test Headline",
                        "subhead": "Test Subhead",
                        "hero_paragraph": "Test hero paragraph content.",
                        "bulleted_features": ["Feature 1", "Feature 2", "Feature 3"],
                        "persona": "Test persona",
                        "use_cases": ["Use case 1", "Use case 2"],
                        "ctas": ["Primary CTA", "Secondary CTA"],
                        "email_subject": "Test email subject",
                        "email_body": "Test email body content.",
                        "social_posts": ["Tweet content", "LinkedIn content"]
                    }''',
                    "role": "assistant"
                }
            }
        ]
    }


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint returns success."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestExtractEndpoint:
    """Test feature extraction endpoint."""
    
    @patch('app.llm_client.LLMClient.chat_completion')
    def test_extract_features_success(self, mock_completion, client, mock_llm_response):
        """Test successful feature extraction."""
        mock_completion.return_value = mock_llm_response
        
        request_data = {
            "product_prompt": "A portable 4K projector with long battery life",
            "max_features": 5
        }
        
        response = client.post("/extract", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "features" in data
        assert len(data["features"]) >= 1
        assert data["features"][0]["name"] == "Portable Design"
    
    def test_extract_features_validation_error(self, client):
        """Test extraction with invalid input."""
        request_data = {
            "product_prompt": "",  # Empty prompt should fail
            "max_features": 5
        }
        
        response = client.post("/extract", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_extract_features_max_features_validation(self, client):
        """Test max_features validation."""
        request_data = {
            "product_prompt": "Valid product description",
            "max_features": 100  # Too high, should fail
        }
        
        response = client.post("/extract", json=request_data)
        assert response.status_code == 422


class TestStorylineEndpoint:
    """Test storyline generation endpoint."""
    
    @patch('app.llm_client.LLMClient.chat_completion')
    def test_generate_storyline_success(self, mock_completion, client, mock_storyline_response):
        """Test successful storyline generation."""
        mock_completion.return_value = mock_storyline_response
        
        request_data = {
            "product_prompt": "A portable 4K projector with long battery life",
            "tone": "friendly",
            "length": "medium"
        }
        
        response = client.post("/storyline", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "headline" in data
        assert "hero_paragraph" in data
        assert "bulleted_features" in data
        assert len(data["bulleted_features"]) >= 3
    
    @patch('app.llm_client.LLMClient.chat_completion')  
    def test_generate_storyline_with_features(self, mock_completion, client, mock_storyline_response):
        """Test storyline generation with provided features."""
        mock_completion.return_value = mock_storyline_response
        
        features = [
            {
                "id": "f1",
                "name": "Portable Design", 
                "type": "benefit",
                "importance_rank": 1,
                "confidence": 0.95,
                "example_phrase": "Fits in your hand"
            }
        ]
        
        request_data = {
            "product_prompt": "A portable 4K projector",
            "features": features,
            "tone": "luxury",
            "length": "long"
        }
        
        response = client.post("/storyline", json=request_data)
        assert response.status_code == 200
    
    def test_generate_storyline_validation_error(self, client):
        """Test storyline generation with invalid input."""
        request_data = {
            "product_prompt": "",  # Empty prompt
            "tone": "friendly"
        }
        
        response = client.post("/storyline", json=request_data)
        assert response.status_code == 422
