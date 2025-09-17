"""
Configuration management using environment variables.
"""

import os
from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    LLM_API_KEY: str = "your-api-key-here"  # Set via environment
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000
    LLM_TOP_P: float = 1.0
    LLM_PRESENCE_PENALTY: float = 0.0
    LLM_FREQUENCY_PENALTY: float = 0.0
    
    # Application Configuration
    APP_NAME: str = "Marketing Storyline Generator"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 3600  # seconds
    
    # Caching
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # seconds
    CACHE_MAX_SIZE: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
