"""
Feature extraction service for product marketing analysis.

This module handles extracting, deduplicating, and ranking marketing features
from product descriptions using LLM APIs.
"""

import json
import logging
from typing import List, Dict, Any, Set
from cachetools import TTLCache
import hashlib

from app.llm_client import LLMClient, LLMError
from app.prompt_templates import PromptTemplates
from app.schemas import MarketingFeature
from app.config import get_settings

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Service for extracting and processing marketing features from product descriptions.
    
    This class handles the complete pipeline from raw product description to 
    structured, ranked marketing features with confidence scores.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize feature extractor.
        
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
    
    def _generate_cache_key(self, product_prompt: str, max_features: int) -> str:
        """Generate cache key for feature extraction request."""
        content = f"{product_prompt}:{max_features}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def extract_features(
        self, 
        product_prompt: str, 
        max_features: int = 10
    ) -> List[MarketingFeature]:
        """
        Extract marketing features from product description.
        
        Args:
            product_prompt: Product description to analyze
            max_features: Maximum number of features to extract
            
        Returns:
            List of MarketingFeature objects, ranked by importance
            
        Raises:
            ValueError: If invalid input parameters
            LLMError: If LLM API call fails
        """
        # Validate inputs
        if not product_prompt.strip():
            raise ValueError("Product prompt cannot be empty")
        if max_features < 1 or max_features > 50:
            raise ValueError("max_features must be between 1 and 50")
        
        # Check cache
        cache_key = self._generate_cache_key(product_prompt, max_features)
        if self.cache and cache_key in self.cache:
            logger.info("Returning cached feature extraction result")
            return self.cache[cache_key]
        
        try:
            # Generate extraction prompt
            prompt = PromptTemplates.get_feature_extraction_prompt(
                product_prompt=product_prompt,
                max_features=max_features
            )
            
            # Call LLM API
            messages = [
                {"role": "system", "content": "You are an expert marketing analyst. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            logger.info(f"Extracting features using LLM for prompt: {product_prompt[:100]}...")
            response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent extraction
                max_tokens=2000
            )
            
            # Parse response
            features_data = await self.llm_client.extract_json_response(response)
            
            # Convert to MarketingFeature objects
            features = []
            for idx, feature_data in enumerate(features_data[:max_features]):
                try:
                    # Ensure required fields with defaults
                    feature_data.setdefault("id", f"f{idx + 1}")
                    feature_data.setdefault("importance_rank", idx + 1)
                    feature_data.setdefault("confidence", 0.8)
                    
                    feature = MarketingFeature(**feature_data)
                    features.append(feature)
                except Exception as e:
                    logger.warning(f"Skipping invalid feature data: {e}")
                    continue
            
            # Deduplicate and rank features
            features = self._deduplicate_features(features)
            features = self._rank_features(features)
            
            # Cache results
            if self.cache:
                self.cache[cache_key] = features
            
            logger.info(f"Successfully extracted {len(features)} features")
            return features
            
        except LLMError as e:
            logger.error(f"LLM API error during feature extraction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during feature extraction: {e}")
            raise LLMError(f"Feature extraction failed: {e}")
    
    def _deduplicate_features(self, features: List[MarketingFeature]) -> List[MarketingFeature]:
        """
        Remove duplicate features based on name similarity.
        
        Args:
            features: List of features to deduplicate
            
        Returns:
            List of unique features
        """
        if not features:
            return features
        
        unique_features = []
        seen_names = set()
        
        for feature in features:
            # Normalize name for comparison
            normalized_name = feature.name.lower().strip()
            
            # Skip if we've seen a very similar name
            if any(self._names_similar(normalized_name, seen) for seen in seen_names):
                logger.debug(f"Skipping duplicate feature: {feature.name}")
                continue
            
            seen_names.add(normalized_name)
            unique_features.append(feature)
        
        logger.info(f"Deduplicated {len(features)} features to {len(unique_features)} unique features")
        return unique_features
    
    def _names_similar(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """
        Check if two feature names are similar enough to be considered duplicates.
        
        Args:
            name1: First feature name
            name2: Second feature name  
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            True if names are similar enough to be duplicates
        """
        # Simple similarity check based on word overlap
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = overlap / union if union > 0 else 0
        return similarity >= threshold
    
    def _rank_features(self, features: List[MarketingFeature]) -> List[MarketingFeature]:
        """
        Re-rank features by marketing importance and confidence.
        
        Args:
            features: List of features to rank
            
        Returns:
            List of features sorted by marketing importance
        """
        # Sort by importance rank (lower number = higher importance)
        # Then by confidence (higher = better) as tie-breaker
        features.sort(key=lambda f: (f.importance_rank, -f.confidence))
        
        # Update importance ranks to be sequential
        for idx, feature in enumerate(features):
            feature.importance_rank = idx + 1
        
        return features
