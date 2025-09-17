"""
FastAPI application for product marketing storyline generation.

This application provides endpoints to extract marketing features from product prompts
and generate comprehensive marketing storylines using LLM APIs.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.schemas import (
    ExtractRequest, ExtractResponse, 
    StorylineRequest, StorylineResponse
)
from app.feature_extractor import FeatureExtractor
from app.story_generator import StoryGenerator
from app.llm_client import LLMClient
from app.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Starting up FastAPI application")
    yield
    logger.info("Shutting down FastAPI application")

app = FastAPI(
    title="Product Marketing Storyline Generator",
    description="Extract marketing features and generate storylines from product descriptions using LLM APIs",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection for components
def get_llm_client() -> LLMClient:
    """Dependency to provide LLM client instance."""
    return LLMClient()

def get_feature_extractor(llm_client: LLMClient = Depends(get_llm_client)) -> FeatureExtractor:
    """Dependency to provide feature extractor instance."""
    return FeatureExtractor(llm_client)

def get_story_generator(llm_client: LLMClient = Depends(get_llm_client)) -> StoryGenerator:
    """Dependency to provide story generator instance."""
    return StoryGenerator(llm_client)

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "marketing-storyline-generator"}

@app.post("/extract", response_model=ExtractResponse)
@limiter.limit("10/minute")
async def extract_features(
    request: Request,
    extract_request: ExtractRequest,
    feature_extractor: FeatureExtractor = Depends(get_feature_extractor)
) -> ExtractResponse:
    """
    Extract structured marketing features from a product description.
    
    Args:
        extract_request: Request containing product prompt and max features limit
        feature_extractor: Injected feature extraction service
        
    Returns:
        ExtractResponse: List of extracted and ranked marketing features
        
    Raises:
        HTTPException: If extraction fails or invalid input provided
    """
    try:
        logger.info(f"Extracting features for product: {extract_request.product_prompt[:100]}...")
        
        features = await feature_extractor.extract_features(
            product_prompt=extract_request.product_prompt,
            max_features=extract_request.max_features
        )
        
        logger.info(f"Successfully extracted {len(features)} features")
        return ExtractResponse(features=features)
        
    except ValueError as e:
        logger.error(f"Validation error in feature extraction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in feature extraction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during feature extraction")

@app.post("/storyline", response_model=StorylineResponse)
@limiter.limit("5/minute")
async def generate_storyline(
    request: Request,
    storyline_request: StorylineRequest,
    story_generator: StoryGenerator = Depends(get_story_generator),
    feature_extractor: FeatureExtractor = Depends(get_feature_extractor)
) -> StorylineResponse:
    """
    Generate a comprehensive marketing storyline package.
    
    Args:
        storyline_request: Request containing product prompt, optional features, tone, length, and audience
        story_generator: Injected story generation service
        feature_extractor: Injected feature extraction service (used if features not provided)
        
    Returns:
        StorylineResponse: Complete marketing package with headlines, copy, CTAs, etc.
        
    Raises:
        HTTPException: If generation fails or invalid input provided
    """
    try:
        logger.info(f"Generating storyline for product: {storyline_request.product_prompt[:100]}...")
        
        # Extract features if not provided
        features = storyline_request.features
        if not features:
            logger.info("No features provided, extracting from product prompt")
            extract_result = await feature_extractor.extract_features(
                product_prompt=storyline_request.product_prompt,
                max_features=10
            )
            features = extract_result
        
        storyline = await story_generator.generate_storyline(
            product_prompt=storyline_request.product_prompt,
            features=features,
            tone=storyline_request.tone,
            length=storyline_request.length,
            audience=storyline_request.audience
        )
        
        logger.info("Successfully generated marketing storyline")
        return storyline
        
    except ValueError as e:
        logger.error(f"Validation error in storyline generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in storyline generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during storyline generation")

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

