"""
Story generation service for creating comprehensive marketing packages.

This module handles generating complete marketing storylines including headlines,
copy, CTAs, and social media content from product features.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from cachetools import TTLCache
import hashlib

from app.llm_client import LLMClient, LLMError
from app.prompt_templates import PromptTemplates
from app.schemas import MarketingFeature, StorylineResponse
from app.config import get_settings

logger = logging.getLogger(__name__)


class StoryGenerator:
    """
    Service for generating comprehensive marketing storylines from product features.
    
    This class creates complete marketing packages including headlines, hero copy,
    features, personas, use cases, CTAs, and social media content.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize story generator.
        
        Args:
            llm_client: LLM client instance for API calls
        """
        self.llm_client = llm_client
        self.settings = get_settings()
        
        # Initialize cache if enabled
        if self.settings.CACHE_ENABLED:
            self.cache = TTLCache(
                maxsize=self.settings.CACHE_MAX_SIZE,
                ttl=self.settings.CACHE_TTL
            )
        else:
            self.cache = None
    
    def _generate_cache_key(
        self, 
        product_prompt: str, 
        features: List[MarketingFeature],
        tone: str,
        length: str,
        audience: Optional[str]
    ) -> str:
        """Generate cache key for storyline generation request."""
        features_str = "|".join([f"{f.name}:{f.type}" for f in features])
        content = f"{product_prompt}:{features_str}:{tone}:{length}:{audience or 'default'}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def generate_storyline(
        self,
        product_prompt: str,
        features: List[MarketingFeature],
        tone: str = "friendly",
        length: str = "medium",
        audience: Optional[str] = None
    ) -> StorylineResponse:
        """
        Generate comprehensive marketing storyline package.
        
        Args:
            product_prompt: Original product description
            features: List of marketing features to incorporate
            tone: Marketing tone (friendly, authoritative, playful, luxury, casual)
            length: Content length (short, medium, long)
            audience: Target audience description
            
        Returns:
            StorylineResponse with complete marketing package
            
        Raises:
            ValueError: If invalid input parameters
            LLMError: If LLM API call fails
        """
        # Validate inputs
        if not product_prompt.strip():
            raise ValueError("Product prompt cannot be empty")
        if not features:
            raise ValueError("At least one feature must be provided")
        
        # Check cache
        cache_key = self._generate_cache_key(product_prompt, features, tone, length, audience)
        if self.cache and cache_key in self.cache:
            logger.info("Returning cached storyline generation result")
            return self.cache[cache_key]
        
        try:
            # Convert features to dict format for prompt
            features_dict = [
                {
                    "name": f.name,
                    "type": f.type,
                    "example_phrase": f.example_phrase,
                    "importance_rank": f.importance_rank
                }
                for f in features
            ]
            
            # Generate storyline prompt
            prompt = PromptTemplates.get_storyline_generation_prompt(
                product_prompt=product_prompt,
                features=features_dict,
                tone=tone,
                length=length,
                audience=audience
            )
            
            # Call LLM API
            messages = [
                {
                    "role": "system", 
                    "content": f"You are an expert marketing copywriter specializing in {tone} content. Always return valid JSON."
                },
                {"role": "user", "content": prompt}
            ]
            
            logger.info(f"Generating storyline with tone='{tone}', length='{length}'")
            response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.8,  # Higher temperature for more creative content
                max_tokens=3000   # More tokens for comprehensive storyline
            )
            
            # Parse response
            storyline_data = await self.llm_client.extract_json_response(response)
            
            # Post-process and validate storyline data
            storyline_data = self._post_process_storyline(storyline_data, tone, length)
            
            # Convert to StorylineResponse object
            storyline = StorylineResponse(**storyline_data)
            
            # Cache results
            if self.cache:
                self.cache[cache_key] = storyline
            
            logger.info("Successfully generated marketing storyline")
            return storyline
            
        except LLMError as e:
            logger.error(f"LLM API error during storyline generation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during storyline generation: {e}")
            raise LLMError(f"Storyline generation failed: {e}")
    
    def _post_process_storyline(
        self, 
        storyline_data: Dict[str, Any], 
        tone: str, 
        length: str
    ) -> Dict[str, Any]:
        """
        Post-process and validate storyline data.
        
        Args:
            storyline_data: Raw storyline data from LLM
            tone: Marketing tone used
            length: Content length used
            
        Returns:
            Processed and validated storyline data
        """
        # Ensure all required fields exist with defaults
        defaults = {
            "headline": "Discover Your Perfect Solution",
            "subhead": "Transform your experience with our innovative product",
            "hero_paragraph": "Our product solves your challenges with innovative features designed for your success.",
            "bulleted_features": ["Key benefit 1", "Key benefit 2", "Key benefit 3"],
            "persona": "Target customers who value quality and innovation",
            "use_cases": ["Primary use case", "Secondary use case"],
            "ctas": ["Get Started Today", "Learn More"],
            "email_subject": "Transform your experience today",
            "email_body": "Discover how our innovative solution can transform your experience.",
            "social_posts": ["Great solution for your needs!", "Transform your workflow with innovation."]
        }
        
        # Apply defaults for missing fields
        for key, default_value in defaults.items():
            if key not in storyline_data or not storyline_data[key]:
                storyline_data[key] = default_value
                logger.warning(f"Applied default value for missing field: {key}")
        
        # Validate and clean content based on length requirements
        storyline_data = self._apply_length_constraints(storyline_data, length)
        
        # Ensure lists have appropriate number of items
        if len(storyline_data["bulleted_features"]) < 3:
            storyline_data["bulleted_features"].extend([
                "Additional benefit" for _ in range(3 - len(storyline_data["bulleted_features"]))
            ])
        
        if len(storyline_data["use_cases"]) < 2:
            storyline_data["use_cases"].extend([
                "Additional use case" for _ in range(2 - len(storyline_data["use_cases"]))
            ])
        
        if len(storyline_data["ctas"]) < 2:
            storyline_data["ctas"].extend([
                "Take Action" for _ in range(2 - len(storyline_data["ctas"]))
            ])
        
        if len(storyline_data["social_posts"]) < 2:
            storyline_data["social_posts"].extend([
                "Check out this amazing product!" for _ in range(2 - len(storyline_data["social_posts"]))
            ])
        
        return storyline_data
    
    def _apply_length_constraints(
        self, 
        storyline_data: Dict[str, Any], 
        length: str
    ) -> Dict[str, Any]:
        """
        Apply length constraints based on specified length parameter.
        
        Args:
            storyline_data: Storyline data to constrain
            length: Target length (short, medium, long)
            
        Returns:
            Length-constrained storyline data
        """
        if length == "short":
            # Truncate content for short format
            if len(storyline_data.get("headline", "")) > 60:
                storyline_data["headline"] = storyline_data["headline"][:60] + "..."
            
            # Limit hero paragraph to ~100 words
            hero_words = storyline_data.get("hero_paragraph", "").split()
            if len(hero_words) > 100:
                storyline_data["hero_paragraph"] = " ".join(hero_words[:100]) + "..."
                
        elif length == "long":
            # Ensure content is comprehensive for long format
            if len(storyline_data.get("hero_paragraph", "").split()) < 150:
                # Content might be too short for long format, but we'll keep it as is
                # The LLM should have generated appropriate length content
                pass
        
        return storyline_data
