"""
Tests for LLM client functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx

from app.llm_client import LLMClient, LLMError
from app.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.LLM_API_KEY = "test-api-key"
    settings.LLM_BASE_URL = "https://api.test.com/v1"
    settings.LLM_MODEL = "gpt-3.5-turbo"
    settings.LLM_TEMPERATURE = 0.7
    settings.LLM_MAX_TOKENS = 1000
    settings.LLM_TOP_P = 1.0
    settings.LLM_PRESENCE_PENALTY = 0.0
    settings.LLM_FREQUENCY_PENALTY = 0.0
    return settings


@pytest.fixture
def llm_client(mock_settings):
    """LLM client instance for testing."""
    return LLMClient(settings=mock_settings)


@pytest.fixture
def sample_api_response():
    """Sample OpenAI-style API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Test response content",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }


class TestLLMClient:
    """Test LLM client functionality."""
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_chat_completion_success(self, mock_post, llm_client, sample_api_response):
        """Test successful chat completion."""
        mock_response = Mock()
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test message"}]
        result = await llm_client.chat_completion(messages)
        
        assert result == sample_api_response
        mock_post.assert_called_once()
        
        # Verify request structure
        call_args = mock_post.call_args
        assert "json" in call_args.kwargs
        request_json = call_args.kwargs["json"]
        assert request_json["model"] == "gpt-3.5-turbo"
        assert request_json["messages"] == messages
    
    @pytest.mark.asyncio
    async def test_chat_completion_mock_response(self):
        """Test chat completion with mock API key (development mode)."""
        # Client with default "your-api-key-here" should return mock response
        client = LLMClient()
        
        messages = [{"role": "user", "content": "extract features"}]
        result = await client.chat_completion(messages)
        
        # Should return mock response structure
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "message" in result["choices"][0]
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_chat_completion_timeout(self, mock_post, llm_client):
        """Test handling of timeout errors."""
        mock_post.side_effect = httpx.TimeoutException("Request timeout")
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(LLMError, match="LLM API request timed out"):
            await llm_client.chat_completion(messages)
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_chat_completion_auth_error(self, mock_post, llm_client):
        """Test handling of authentication errors."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.side_effect = httpx.HTTPStatusError(
            "Auth error", request=Mock(), response=mock_response
        )
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(LLMError, match="Invalid API key or authentication failed"):
            await llm_client.chat_completion(messages)
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_chat_completion_rate_limit(self, mock_post, llm_client):
        """Test handling of rate limit errors."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_post.side_effect = httpx.HTTPStatusError(
            "Rate limit", request=Mock(), response=mock_response
        )
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(LLMError, match="Rate limit exceeded"):
            await llm_client.chat_completion(messages)
    
    @pytest.mark.asyncio
    async def test_extract_json_response_success(self, llm_client):
        """Test successful JSON extraction from response."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"key": "value"}',
                        "role": "assistant"
                    }
                }
            ]
        }
        
        result = await llm_client.extract_json_response(response)
        assert result == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_extract_json_response_with_markdown(self, llm_client):
        """Test JSON extraction from markdown-wrapped response."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": '``````',
                        "role": "assistant"
                    }
                }
            ]
        }
        
        result = await llm_client.extract_json_response(response)
        assert result == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_extract_json_response_invalid(self, llm_client):
        """Test handling of invalid JSON in response."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Not valid JSON",
                        "role": "assistant"
                    }
                }
            ]
        }
        
        with pytest.raises(LLMError, match="Could not parse JSON from LLM response"):
            await llm_client.extract_json_response(response)
