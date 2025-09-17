"""
Tests for feature extraction service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from app.feature_extractor import FeatureExtractor
from app.llm_client import LLMClient, LLMError
from app.schemas import MarketingFeature


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = Mock(spec=LLMClient)
    client.chat_completion = AsyncMock()
    client.extract_json_response = AsyncMock()
    return client


@pytest.fixture
def feature_extractor(mock_llm_client):
    """Feature extractor instance for testing."""
    return FeatureExtractor(mock_llm_client)


@pytest.fixture
def sample_features_response():
    """Sample feature extraction response."""
    return [
        {
            "id": "f1",
            "name": "Portable Design",
            "type": "benefit", 
            "importance_rank": 1,
            "confidence": 0.95,
            "example_phrase": "Fits in your hand"
        },
        {
            "id": "f2", 
            "name": "4K Resolution",
            "type": "spec",
            "importance_rank": 2,
            "confidence": 0.90,
            "example_phrase": "Crystal clear 4K display"
        }
    ]


class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    @pytest.mark.asyncio
    async def test_extract_features_success(self, feature_extractor, mock_llm_client, sample_features_response):
        """Test successful feature extraction."""
        mock_llm_client.extract_json_response.return_value = sample_features_response
        
        result = await feature_extractor.extract_features(
            product_prompt="A portable 4K projector",
            max_features=5
        )
        
        assert len(result) == 2
        assert isinstance(result[0], MarketingFeature)
        assert result[0].name == "Portable Design"
        assert result[0].type == "benefit"
        
        # Verify LLM was called
        mock_llm_client.chat_completion.assert_called_once()
        mock_llm_client.extract_json_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_features_empty_prompt(self, feature_extractor):
        """Test extraction with empty prompt."""
        with pytest.raises(ValueError, match="Product prompt cannot be empty"):
            await feature_extractor.extract_features("", max_features=5)
    
    @pytest.mark.asyncio
    async def test_extract_features_invalid_max_features(self, feature_extractor):
        """Test extraction with invalid max_features."""
        with pytest.raises(ValueError, match="max_features must be between 1 and 50"):
            await feature_extractor.extract_features("Valid prompt", max_features=0)
        
        with pytest.raises(ValueError, match="max_features must be between 1 and 50"):
            await feature_extractor.extract_features("Valid prompt", max_features=100)
    
    @pytest.mark.asyncio
    async def test_extract_features_llm_error(self, feature_extractor, mock_llm_client):
        """Test handling of LLM errors."""
        mock_llm_client.chat_completion.side_effect = LLMError("API error")
        
        with pytest.raises(LLMError, match="API error"):
            await feature_extractor.extract_features("Valid prompt", max_features=5)
    
    def test_deduplicate_features(self, feature_extractor):
        """Test feature deduplication."""
        features = [
            MarketingFeature(
                id="f1", name="Portable Design", type="benefit", 
                importance_rank=1, confidence=0.9, example_phrase="Test"
            ),
            MarketingFeature(
                id="f2", name="Portable", type="benefit", 
                importance_rank=2, confidence=0.8, example_phrase="Test"  # Similar to first
            ),
            MarketingFeature(
                id="f3", name="4K Display", type="spec", 
                importance_rank=3, confidence=0.9, example_phrase="Test"
            )
        ]
        
        result = feature_extractor._deduplicate_features(features)
        
        # Should remove the duplicate "Portable" feature
        assert len(result) == 2
        assert result[0].name == "Portable Design"
        assert result[1].name == "4K Display"
    
    def test_names_similar(self, feature_extractor):
        """Test name similarity detection."""
        assert feature_extractor._names_similar("portable design", "portable")
        assert feature_extractor._names_similar("4k display", "4k resolution")
        assert not feature_extractor._names_similar("portable", "wireless")
    
    def test_rank_features(self, feature_extractor):
        """Test feature ranking."""
        features = [
            MarketingFeature(
                id="f1", name="Feature 1", type="benefit", 
                importance_rank=3, confidence=0.7, example_phrase="Test"
            ),
            MarketingFeature(
                id="f2", name="Feature 2", type="spec", 
                importance_rank=1, confidence=0.9, example_phrase="Test"
            ),
            MarketingFeature(
                id="f3", name="Feature 3", type="benefit", 
                importance_rank=2, confidence=0.8, example_phrase="Test"
            )
        ]
        
        result = feature_extractor._rank_features(features)
        
        # Should be ranked by importance (lower rank number = higher importance)
        assert result[0].importance_rank == 1
        assert result[1].importance_rank == 2  
        assert result[2].importance_rank == 3
        assert result[0].name == "Feature 2"  # Originally rank 1
