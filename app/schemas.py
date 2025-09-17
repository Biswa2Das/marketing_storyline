"""
Pydantic schemas for request/response validation and serialization.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class FeatureType(str, Enum):
    """Enumeration of marketing feature types."""
    BENEFIT = "benefit"
    SPEC = "spec"
    USE_CASE = "use-case"
    AUDIENCE = "audience"


class ToneType(str, Enum):
    """Enumeration of marketing tone types."""
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    PLAYFUL = "playful"
    LUXURY = "luxury"
    CASUAL = "casual"


class LengthType(str, Enum):
    """Enumeration of content length types."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class MarketingFeature(BaseModel):
    """Individual marketing feature with metadata."""
    
    id: str = Field(..., description="Unique identifier for the feature")
    name: str = Field(..., description="Human-readable feature name")
    type: FeatureType = Field(..., description="Type of marketing feature")
    importance_rank: int = Field(..., ge=1, description="Ranking by marketing importance (1 is highest)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for feature extraction")
    example_phrase: str = Field(..., description="Example marketing phrase using this feature")

    class Config:
        use_enum_values = True


class ExtractRequest(BaseModel):
    """Request schema for feature extraction endpoint."""
    
    product_prompt: str = Field(
        ..., 
        min_length=10,
        max_length=5000,
        description="Product description or prompt to extract features from"
    )
    max_features: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of features to extract"
    )

    @validator('product_prompt')
    def validate_product_prompt(cls, v):
        if not v.strip():
            raise ValueError('Product prompt cannot be empty or only whitespace')
        return v.strip()


class ExtractResponse(BaseModel):
    """Response schema for feature extraction endpoint."""
    
    features: List[MarketingFeature] = Field(..., description="List of extracted marketing features")


class StorylineRequest(BaseModel):
    """Request schema for storyline generation endpoint."""
    
    product_prompt: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Product description or prompt"
    )
    features: Optional[List[MarketingFeature]] = Field(
        default=None,
        description="Optional pre-extracted features (if not provided, will be extracted automatically)"
    )
    tone: ToneType = Field(
        default=ToneType.FRIENDLY,
        description="Marketing tone for the generated content"
    )
    length: LengthType = Field(
        default=LengthType.MEDIUM,
        description="Target length for generated content"
    )
    audience: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Target audience description"
    )

    @validator('product_prompt')
    def validate_product_prompt(cls, v):
        if not v.strip():
            raise ValueError('Product prompt cannot be empty or only whitespace')
        return v.strip()

    class Config:
        use_enum_values = True


class StorylineResponse(BaseModel):
    """Response schema for storyline generation endpoint."""
    
    headline: str = Field(..., description="Main marketing headline")
    subhead: str = Field(..., description="Supporting subheadline")
    hero_paragraph: str = Field(..., description="Hero section paragraph text")
    bulleted_features: List[str] = Field(..., description="List of key features as bullet points")
    persona: str = Field(..., description="Target customer persona")
    use_cases: List[str] = Field(..., description="List of primary use cases")
    ctas: List[str] = Field(..., description="Call-to-action texts (primary and secondary)")
    email_subject: str = Field(..., description="Email marketing subject line")
    email_body: str = Field(..., description="Email marketing body text")
    social_posts: List[str] = Field(..., description="Social media posts (various sizes)")


class LLMRequest(BaseModel):
    """Internal schema for LLM API requests."""
    
    model: str = Field(..., description="LLM model name")
    messages: List[dict] = Field(..., description="Chat messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
