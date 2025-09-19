# description.py
import json
import logging
from typing import Dict, Any, List, Optional

import requests

# --- Configuration ---
class Config:
    OLLAMA_BASE_URL = "http://localhost:11434"   # native endpoint (no /v1) [web:38]
    MODEL = "llama2"
    REQUEST_TIMEOUT = 180

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- I/O helpers ---
def load_storyline_json(path: str = "storyline.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_scenes_json(scenes: List[Dict[str, Any]], path: str = "scenes.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scenes": scenes}, f, ensure_ascii=False, indent=2)

# --- Scene Generator using /api/generate ---
class SceneGenerator:
    def __init__(self, config: Config):
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.model = config.MODEL
        self.timeout = config.REQUEST_TIMEOUT
        # quick health check
        r = requests.get(f"{self.base_url}/", timeout=5)
        if r.status_code == 200:
            logging.info(f"✅ Connected to Ollama at {self.base_url} using model: {self.model}")
        else:
            raise RuntimeError(f"Ollama not reachable at {self.base_url}, status {r.status_code}")

    def _build_prompt(self, storyline: Dict[str, Any], num_scenes: int, video_length: str) -> str:
        tagline = storyline.get("tagline") or storyline.get("headline") or ""
        narrative = storyline.get("narrative") or storyline.get("story") or ""
        storyline_view = {"tagline": tagline, "narrative": narrative}
        storyline_text = json.dumps(storyline_view, ensure_ascii=False, indent=2)
        return f"""
Here is the marketing storyline (tagline + narrative):
{storyline_text}

Task: Generate exactly {num_scenes} scene descriptions for a {video_length} commercial.

Return ONLY a valid JSON object with this exact shape:
{{
  "scenes": [
    {{
      "scene": <integer starting at 1>,
      "visuals": "<detailed setting, characters, action, product focus, mood, lighting>",
      "camera": "<specific shot/angle/movement>",
      "on_screen_text": "<3-5 word overlay reinforcing a key benefit>",
      "sound": "<brief music/VO/SFX description>"
    }}
  ]
}}

Rules:
- Respond with pure JSON only. No markdown, no commentary, no backticks.
"""

    def generate_scenes(
        self,
        storyline: Dict[str, Any],
        num_scenes: int = 3,
        video_length: str = "15-second",
        temperature: float = 0.7,
    ) -> Optional[List[Dict[str, Any]]]:
        prompt = self._build_prompt(storyline, num_scenes, video_length)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # single JSON response [web:38]
            "options": {
                "temperature": temperature,
                "num_predict": 512,
                "top_p": 0.9,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logging.error(f"❌ Ollama /api/generate error: {resp.status_code} {resp.text[:300]}")
                return None

            data = resp.json() or {}
            text = data.get("response", "") or ""

            # Extract JSON block from text just in case model adds extra tokens
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]

            scene_data = json.loads(text)
            if isinstance(scene_data, dict) and isinstance(scene_data.get("scenes"), list):
                logging.info(f"✅ Generated {len(scene_data['scenes'])} scenes.")
                return scene_data["scenes"]
            logging.warning("⚠ Output JSON not in expected format {'scenes': [...]} — returning None.")
            return None

        except requests.RequestException as e:
            logging.error(f"❌ Request failed: {e}")
            return None
        except json.JSONDecodeError:
            logging.error("❌ Failed to decode JSON from model output.")
            return None

# --- Main (optional CLI test) ---
if __name__ == "__main__":
    cfg = Config()
    gen = SceneGenerator(cfg)
    storyline = load_storyline_json("storyline.json")
    scenes = gen.generate_scenes(storyline, num_scenes=4, video_length="20-second")
    if scenes:
        save_scenes_json(scenes, "scenes.json")
        print(json.dumps({"scenes": scenes}, ensure_ascii=False, indent=2))
    else:
        print("No scenes generated. Check logs.")
