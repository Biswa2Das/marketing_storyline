"""
Template management for LLM prompts.

This module contains all prompt templates used for feature extraction
and storyline generation, organized for easy maintenance and customization.
"""

from typing import Dict, List, Any
from string import Template


class PromptTemplates:
    """Container for all prompt templates used in the application."""
    
    # Feature extraction prompt template
    FEATURE_EXTRACTION_TEMPLATE = Template("""
You are an expert marketing analyst. Extract key marketing features from the following product description.

Product Description:
${product_prompt}

Instructions:
1. Identify marketing features that would be valuable for promoting this product
2. Classify each feature as: benefit, spec, use-case, or audience
3. Rank features by marketing importance (1 = most important)
4. Provide confidence score (0.0 to 1.0) for each extraction
5. Create an example marketing phrase for each feature
6. Return exactly ${max_features} features maximum
7. Ensure each feature has a unique ID (f1, f2, f3, etc.)

Return the results as a JSON array with this exact structure:
[
    {
        "id": "f1",
        "name": "Feature Name",
        "type": "benefit|spec|use-case|audience",
        "importance_rank": 1,
        "confidence": 0.95,
        "example_phrase": "Marketing phrase using this feature"
    }
]

Focus on features that would be most compelling to potential customers and easiest to market effectively.
""")

    # Storyline generation prompt template  
    STORYLINE_GENERATION_TEMPLATE = Template("""
You are an expert marketing copywriter. Create a comprehensive marketing storyline package for the following product.

Product Description:
${product_prompt}

Key Features to Highlight:
${features_list}

Marketing Parameters:
- Tone: ${tone}
- Content Length: ${length}
- Target Audience: ${audience}

Instructions:
Create a complete marketing package that tells a compelling story following this narrative arc:
1. Problem identification (what challenge does this solve?)
2. Solution presentation (how does this product help?)
3. Value demonstration (why is this better?)
4. Social proof/examples (who would use this and how?)

Generate content with the specified tone (${tone}) and length (${length}). 
${length_guidance}
${tone_guidance}

Return the results as JSON with this exact structure:
{
    "headline": "Compelling main headline (under 80 characters)",
    "subhead": "Supporting subheadline that adds context",
    "hero_paragraph": "Hero section paragraph that tells the problem→solution→value story",
    "bulleted_features": ["Feature 1 as benefit statement", "Feature 2 as benefit statement", "Feature 3 as benefit statement"],
    "persona": "Primary target customer persona description",
    "use_cases": ["Use case 1", "Use case 2", "Use case 3"],
    "ctas": ["Primary CTA text", "Secondary CTA text"],
    "email_subject": "Email subject line (under 50 characters)",
    "email_body": "Email body text following the same narrative arc",
    "social_posts": ["Tweet-sized post (under 280 chars)", "LinkedIn post (longer form)"]
}

Make the content compelling, specific, and action-oriented. Focus on customer benefits rather than just features.
""")

    # Length-specific guidance
    LENGTH_GUIDANCE = {
        "short": "Keep all content concise and punchy. Headlines under 60 chars, paragraphs under 100 words.",
        "medium": "Use moderate length content. Headlines 60-80 chars, paragraphs 100-200 words.",
        "long": "Create detailed, comprehensive content. Headlines can be longer, paragraphs 200+ words with rich detail."
    }
    
    # Tone-specific guidance
    TONE_GUIDANCE = {
        "friendly": "Use warm, approachable language. Include conversational elements and focus on helpfulness.",
        "authoritative": "Use professional, expert language. Focus on credibility, data, and proven results.",
        "playful": "Use fun, creative language with personality. Include humor and engaging metaphors where appropriate.",
        "luxury": "Use sophisticated, premium language. Focus on exclusivity, quality, and elevated experiences.",
        "casual": "Use relaxed, informal language. Write as you would speak to a friend, avoiding jargon."
    }

    @classmethod
    def get_feature_extraction_prompt(
        cls, 
        product_prompt: str, 
        max_features: int = 10
    ) -> str:
        """
        Generate feature extraction prompt.
        
        Args:
            product_prompt: Product description to analyze
            max_features: Maximum number of features to extract
            
        Returns:
            Formatted prompt string
        """
        return cls.FEATURE_EXTRACTION_TEMPLATE.substitute(
            product_prompt=product_prompt.strip(),
            max_features=max_features
        )
    
    @classmethod
    def get_storyline_generation_prompt(
        cls,
        product_prompt: str,
        features: List[Dict[str, Any]],
        tone: str = "friendly",
        length: str = "medium", 
        audience: str = "general consumers"
    ) -> str:
        """
        Generate storyline generation prompt.
        
        Args:
            product_prompt: Original product description
            features: List of extracted features
            tone: Marketing tone to use
            length: Target content length
            audience: Target audience description
            
        Returns:
            Formatted prompt string
        """
        # Format features for the prompt
        features_list = "\n".join([
            f"- {feature.get('name', 'Unknown')}: {feature.get('example_phrase', 'No example')}"
            for feature in features
        ])
        
        # Get guidance based on parameters
        length_guidance = cls.LENGTH_GUIDANCE.get(length, cls.LENGTH_GUIDANCE["medium"])
        tone_guidance = cls.TONE_GUIDANCE.get(tone, cls.TONE_GUIDANCE["friendly"])
        
        return cls.STORYLINE_GENERATION_TEMPLATE.substitute(
            product_prompt=product_prompt.strip(),
            features_list=features_list,
            tone=tone,
            length=length,
            audience=audience or "general consumers",
            length_guidance=length_guidance,
            tone_guidance=tone_guidance
        )
