# description.py
import json
import logging
from typing import Dict, Any, List, Optional

from huggingface_hub import InferenceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_default_schema() -> Dict[str, Any]:
    # JSON Schema for strict scene outputs (name/strict per HF structured outputs) [web:191]
    return {
        "name": "Scenes",
        "schema": {
            "type": "object",
            "properties": {
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene": {"type": "integer"},
                            "visuals": {"type": "string"},
                            "camera": {"type": "string"},
                            "on_screen_text": {"type": "string"},
                            "sound": {"type": "string"},
                        },
                        "required": ["scene", "visuals", "camera", "on_screen_text", "sound"],
                    },
                }
            },
            "required": ["scenes"],
            "additionalProperties": False,
        },
        "strict": True,
    }

class SceneGenerator:
    def __init__(self, hf_token: str, model: str = "meta-llama/Llama-3.1-8B-Instruct", temperature: float = 0.7):
        self.client = InferenceClient(provider="auto", api_key=hf_token)  # Providers/Endpoints supported [web:195]
        self.model = model
        self.temperature = temperature
        logging.info(f"✅ Using HF model: {self.model}")

    def _build_prompt(self, storyline: Dict[str, Any], num_scenes: int, video_length: str) -> str:
        tagline = storyline.get("tagline") or storyline.get("headline") or ""
        narrative = storyline.get("narrative") or storyline.get("story") or ""
        storyline_view = {"tagline": tagline, "narrative": narrative}
        story_txt = json.dumps(storyline_view, ensure_ascii=False, indent=2)
        return f"""
Here is the marketing storyline (tagline + narrative):
{story_txt}

Generate exactly {num_scenes} scene descriptions for a {video_length} commercial.

Rules:
- Return ONLY a valid JSON object conforming to the provided JSON Schema.
- No commentary, no markdown, no backticks.
"""

    def generate_scenes(
        self,
        storyline: Dict[str, Any],
        num_scenes: int = 3,
        video_length: str = "15-second",
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        system_prompt = "You are a creative director. Respond ONLY with JSON matching the schema."  # [web:191]
        user_prompt = self._build_prompt(storyline, num_scenes, video_length)

        # Default schema if none provided
        schema = json_schema or build_default_schema()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_schema", "json_schema": schema},  # strict JSON via HF [web:191]
            )
            text = resp.choices[0].message.content or ""
            data = json.loads(text)

            if isinstance(data, dict) and isinstance(data.get("scenes"), list):
                logging.info(f"✅ Generated {len(data['scenes'])} scenes.")
                return data["scenes"]
            logging.warning("⚠ Output did not match {'scenes':[...]} shape.")
            return None
        except Exception as e:
            logging.error(f"❌ HF structured output error: {e}")
            return None
