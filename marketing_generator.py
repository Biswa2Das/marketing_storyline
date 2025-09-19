# marketing_generator.py
import requests
import time
import json
from typing import Dict, Any, List, Optional


class RobustMarketingGenerator:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.ollama_api_url = base_url
        # Prefer installed llama2 first; add small, reliable fallbacks
        self.preferred_models: List[str] = ["llama2", "llama3.2:1b"]
        self.default_options: Dict[str, Any] = {
            "num_predict": 320,   # safe token budget
            "temperature": 0.7,
            "top_p": 0.9,
        }

    # -------- Service / model utilities --------
    def check_ollama_service(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_api_url}/", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.ollama_api_url}/api/tags", timeout=15)
            if r.status_code == 200:
                payload = r.json() or {}
                models = payload.get("models", []) or []
                names = []
                for m in models:
                    name = m.get("name") or m.get("model")
                    if name:
                        names.append(name)
                return names
            return []
        except requests.RequestException:
            return []

    def pull_model(self, name: str, timeout_sec: int = 900) -> bool:
        try:
            r = requests.post(
                f"{self.ollama_api_url}/api/pull",
                json={"name": name},
                timeout=timeout_sec,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    # -------- Prompting / generation --------
    def _build_prompt(self, product_input: str) -> str:
        return f"""You are a creative marketing expert. Based on the following product details, create:

1. A short tagline (10–15 words maximum)
2. A full marketing narrative (100–150 words)

Product Details: {product_input}

Format your response exactly as:
TAGLINE: <tagline>
NARRATIVE: <narrative>
"""

    def _generate_once(self, product_input: str, model_name: str) -> Dict[str, Any]:
        payload = {
            "model": model_name,
            "prompt": self._build_prompt(product_input),
            "stream": False,               # return a single JSON object
            "options": self.default_options,
        }
        try:
            r = requests.post(
                f"{self.ollama_api_url}/api/generate",
                json=payload,
                timeout=120,
            )
            if r.status_code != 200:
                return {
                    "success": False,
                    "error": f"API error {r.status_code}",
                    "status": r.status_code,
                }

            data = r.json() or {}
            content = data.get("response", "") or ""

            # Parse structured output
            tagline, narrative = "", ""
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("TAGLINE:"):
                    tagline = line[len("TAGLINE:"):].strip()
                elif line.startswith("NARRATIVE:"):
                    narrative = line[len("NARRATIVE:"):].strip()

            # Fallback parsing
            if not tagline or not narrative:
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                if lines:
                    tagline = tagline or lines[0][:120]
                    narrative = narrative or " ".join(lines[1:])[:1500]

            # Enforce lengths
            tagline_words = tagline.split()
            if len(tagline_words) > 15:
                tagline = " ".join(tagline_words[:15])
            if len(tagline_words) < 10:
                tagline = " ".join((tagline_words + [""] * (10 - len(tagline_words))))[:120].strip()

            narrative_words = narrative.split()
            if len(narrative_words) > 150:
                narrative = " ".join(narrative_words[:150])
            elif len(narrative_words) < 100:
                narrative = " ".join(narrative_words + [""] * (100 - len(narrative_words))).strip()

            return {
                "success": True,
                "tagline": tagline,
                "narrative": narrative,
                "model": model_name,
            }
        except requests.RequestException as e:
            return {"success": False, "error": f"API request failed: {e}"}

    def _resolve_models(self) -> List[str]:
        local_models = self.list_models()

        # Pull small fallback if preferred not present
        for pref in self.preferred_models:
            if pref not in local_models and pref == "llama3.2:1b":
                # Try to pull a small reliable fallback quietly
                if self.pull_model(pref):
                    time.sleep(2)
                    local_models = self.list_models()

        # Build try order: preferred (available) first, then any remaining locals
        try_order: List[str] = []
        for m in self.preferred_models:
            if m in local_models and m not in try_order:
                try_order.append(m)
        for m in local_models:
            if m not in try_order:
                try_order.append(m)
        return try_order

    # -------- Public API: pure functions for UI integration --------
    def generate_storyline(self, product_input: str) -> Dict[str, Any]:
        """
        Pure function used by UIs. Returns a dict:
        {
          "success": bool,
          "tagline": str,
          "narrative": str,
          "model": str,
          "error": Optional[str]
        }
        """
        if not self.check_ollama_service():
            return {
                "success": False,
                "error": "Ollama service not running. Start it and retry.",
            }

        try_order = self._resolve_models()
        if not try_order:
            return {
                "success": False,
                "error": "No local models available. Pull a model (e.g., 'ollama pull llama3.2:1b').",
            }

        # Attempt generation; retry on server errors
        for model in try_order:
            res = self._generate_once(product_input, model)
            if res.get("success"):
                return res
            status = res.get("status")
            if status == 500:
                # try next model
                continue
        return {
            "success": False,
            "error": "All generation attempts failed with available models.",
        }


# ---------- Convenience helpers for saving/loading ----------
def save_storyline_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Save the generated storyline payload to a JSON file that description.py can load.
    """
    safe = {
        "tagline": payload.get("tagline", ""),
        "narrative": payload.get("narrative", ""),
        "model": payload.get("model", ""),
        "success": bool(payload.get("success", False)),
        "error": payload.get("error"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)


def load_storyline_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Optional CLI usage ----------
if __name__ == "__main__":
    generator = RobustMarketingGenerator()
    product = "Smartwatch with AI health tracking, 7-day battery life, and waterproof design."
    result = generator.generate_storyline(product)

    if result.get("success"):
        print("\nTAGLINE:\n ", result["tagline"])
        print("\nNARRATIVE:\n ", result["narrative"])
        print("\nMODEL:\n ", result["model"])
        save_storyline_json("storyline.json", result)
        print("\nSaved storyline.json")
    else:
        print(f"\nGeneration failed: {result.get('error')}")
