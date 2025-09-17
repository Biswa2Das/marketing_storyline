"""
LLM client abstraction for OpenAI-compatible APIs.

This module provides a thin, configurable wrapper around LLM API calls
with proper error handling and environment-based configuration.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
import httpx
from app.config import get_settings
from app.schemas import LLMRequest

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


class LLMClient:
    """
    OpenAI-compatible LLM client with configurable parameters.
    
    This client abstracts LLM API calls and provides a consistent interface
    for different LLM providers that follow OpenAI's API format.
    """
    
    def __init__(self, settings: Optional[Any] = None):
        """
        Initialize LLM client with configuration.
        
        Args:
            settings: Optional settings object, defaults to app settings
        """
        self.settings = settings or get_settings()
        self.base_url = self.settings.LLM_BASE_URL
        self.api_key = self.settings.LLM_API_KEY
        self.default_model = self.settings.LLM_MODEL
        
        # Validate API key
        if not self.api_key or self.api_key == "your-api-key-here":
            logger.warning("LLM API key not properly configured. Set LLM_API_KEY environment variable.")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate chat completion using LLM API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name (defaults to configured model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            presence_penalty: Presence penalty parameter
            frequency_penalty: Frequency penalty parameter
            
        Returns:
            Dict containing the API response
            
        Raises:
            LLMError: If API call fails or returns invalid response
        """
        # Use provided parameters or fall back to defaults
        request_data = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.LLM_TEMPERATURE,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.LLM_MAX_TOKENS,
            "top_p": top_p if top_p is not None else self.settings.LLM_TOP_P,
            "presence_penalty": presence_penalty if presence_penalty is not None else self.settings.LLM_PRESENCE_PENALTY,
            "frequency_penalty": frequency_penalty if frequency_penalty is not None else self.settings.LLM_FREQUENCY_PENALTY
        }
        
        # Validate request data
        try:
            LLMRequest(**request_data)
        except Exception as e:
            raise LLMError(f"Invalid LLM request parameters: {e}")
        
        # For development/testing, return mock response if API key not configured
        if not self.api_key or self.api_key == "your-api-key-here":
            logger.warning("Using mock LLM response - configure LLM_API_KEY for real API calls")
            return self._mock_response(messages)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=request_data
                )
                
                response.raise_for_status()
                return response.json()
                
        except httpx.TimeoutException:
            raise LLMError("LLM API request timed out")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise LLMError("Invalid API key or authentication failed")
            elif e.response.status_code == 429:
                raise LLMError("Rate limit exceeded")
            else:
                raise LLMError(f"LLM API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise LLMError(f"Unexpected error calling LLM API: {e}")
    
    def _mock_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate mock response for development/testing.
        
        Args:
            messages: Input messages to generate appropriate mock response
            
        Returns:
            Mock API response in OpenAI format
        """
        # Analyze the last message to determine response type
        last_message = messages[-1].get("content", "").lower()
        
        if "extract" in last_message or "features" in last_message:
            mock_content = """[
    {
        "id": "f1",
        "name": "Portable Design",
        "type": "benefit",
        "importance_rank": 1,
        "confidence": 0.95,
        "example_phrase": "Fits in your hand for presentations anywhere"
    },
    {
        "id": "f2", 
        "name": "4K Streaming",
        "type": "spec",
        "importance_rank": 2,
        "confidence": 0.90,
        "example_phrase": "Crystal-clear 4K resolution for professional presentations"
    },
    {
        "id": "f3",
        "name": "Long Battery Life",
        "type": "benefit", 
        "importance_rank": 3,
        "confidence": 0.88,
        "example_phrase": "4-hour battery lasts through the longest meetings"
    }
]"""
        else:
            mock_content = """{
    "headline": "Present Anywhere, Anytime - The Ultimate Portable 4K Projector",
    "subhead": "Transform any space into your presentation room with our palm-sized powerhouse",
    "hero_paragraph": "Gone are the days of being tethered to conference rooms and bulky equipment. Our revolutionary portable projector delivers stunning 4K visuals wherever your work takes you. Whether you're a digital nomad presenting from a coffee shop or a teacher bringing lessons to life, this pocket-sized projector adapts to your world.",
    "bulleted_features": [
        "Fits in your hand for presentations anywhere",
        "Crystal-clear 4K resolution for professional quality",
        "4-hour battery lasts through the longest meetings",
        "Auto keystone correction for perfect alignment",
        "Built-in speakers eliminate extra equipment"
    ],
    "persona": "Tech-savvy professionals and educators who value mobility and quality",
    "use_cases": [
        "Remote work presentations from any location",
        "Interactive classroom teaching without fixed equipment",
        "Client meetings in unconventional spaces",
        "Travel presentations without venue dependencies"
    ],
    "ctas": ["Get Your Portable Projector Today", "Start Presenting Anywhere"],
    "email_subject": "Break free from conference rooms - Present anywhere with 4K clarity",
    "email_body": "Imagine never having to say 'sorry, I can't present there' again. Our new portable 4K projector fits in your hand but delivers boardroom-quality visuals anywhere you go. Perfect for digital nomads, traveling professionals, and innovative educators who refuse to be limited by location.",
    "social_posts": [
        "Conference room or coffee shop? Doesn't matter anymore. 4K presentations anywhere. #PortableProjector #DigitalNomad",
        "Teaching just got more flexible. Bring stunning 4K visuals to any classroom - no installation required. Perfect for educators who think outside the classroom walls. #EdTech #PortableLearning"
    ]
}"""
        
        return {
            "choices": [
                {
                    "message": {
                        "content": mock_content,
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
        }
    
    async def extract_json_response(self, response: Dict[str, Any]) -> Any:
        """
        Extract and parse JSON content from LLM response.
        
        Args:
            response: Raw LLM API response
            
        Returns:
            Parsed JSON object from response content
            
        Raises:
            LLMError: If response cannot be parsed as JSON
        """
        try:
            content = response["choices"][0]["message"]["content"]
            
            # Try to find JSON in the response
            content = content.strip()
            if content.startswith("```
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError(f"Could not parse JSON from LLM response: {e}")
