import json
from typing import Dict, Any, List
from groq import Groq

def build_default_schema() -> Dict[str, Any]:
    # Example JSON schema for scenes; adjust to your exact structure
    return {
        "type": "object",
        "properties": {
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string"},
                        "shot_type": {"type": "string"},
                        "visuals": {"type": "string"},
                        "voiceover": {"type": "string"},
                        "on_screen_text": {"type": "string"},
                        "duration_sec": {"type": "number"},
                        "cta": {"type": "string"},
                    },
                    "required": ["id", "title", "visuals", "voiceover", "duration_sec"],
                },
            }
        },
        "required": ["scenes"],
    }

class SceneGenerator:
    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.7):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.temperature = temperature

    def _build_scene_prompt(
        self,
        storyline: Dict[str, str],
        num_scenes: int,
        video_length: str,
        json_schema: Dict[str, Any],
    ) -> str:
        schema_str = json.dumps(json_schema, indent=2)
        return f"""You are a marketing director and storyboard artist.

Create a scene breakdown for a {video_length} vertical ad based on the following storyline.
Return a STRICT JSON object that validates against the provided JSON Schema. Do not include any text outside JSON.

Storyline:
Tagline: {storyline.get("tagline","")}
Narrative: {storyline.get("narrative","")}

Requirements:
- Exactly {num_scenes} scenes.
- Each scene should include: id, title, shot_type, visuals, voiceover, on_screen_text, duration_sec, cta (if applicable).
- Total time budget should roughly match the {video_length}.
- Keep language concise and production-ready.

JSON Schema:
{schema_str}
"""

    def generate_scenes(
        self,
        storyline: Dict[str, str],
        num_scenes: int,
        video_length: str,
        json_schema: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        prompt = self._build_scene_prompt(storyline, num_scenes, video_length, json_schema)

        # Use JSON-style prompting. If desired, you can add a JSON mode hint in the system message.
        messages = [
            {"role": "system", "content": "Return only a valid JSON object that conforms to the schema. No extra text."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            text = resp.choices[0].message.content or ""
            # Best-effort JSON extraction/parsing
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end+1]
            data = json.loads(text)
            scenes = data.get("scenes", [])
            return scenes
        except Exception as e:
            # Optionally log e
            return []

