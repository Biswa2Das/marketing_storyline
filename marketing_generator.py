# marketing_generator.py
import json
from typing import Dict, Any
from huggingface_hub import InferenceClient

# Generate a structured storyline (tagline + narrative) using a chat prompt template. [web:195]
def _build_prompt(product_input: str) -> str:
    return f"""You are a creative marketing expert. Based on the following product details, create:

1. A short tagline (10–15 words maximum)
2. A full marketing narrative (100–150 words)

Product Details: {product_input}

Format your response exactly as:
TAGLINE: <tagline>
NARRATIVE: <narrative>
"""

def generate_storyline(product_input: str, hf_token: str, model: str = "meta-llama/Llama-3.1-8B-Instruct") -> Dict[str, Any]:
    client = InferenceClient(provider="auto", api_key=hf_token)  # HF Inference Providers [web:195]
    prompt = _build_prompt(product_input)

    messages = [
        {"role": "system", "content": "Return only the requested sections."},
        {"role": "user", "content": prompt},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )  # standard chat completion; formatting enforced via prompt [web:195]
        text = resp.choices[0].message.content or ""

        tagline, narrative = "", ""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("TAGLINE:"):
                tagline = line[len("TAGLINE:"):].strip()
            elif line.startswith("NARRATIVE:"):
                narrative = line[len("NARRATIVE:"):].strip()

        # Fallback parsing
        if not tagline or not narrative:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                tagline = tagline or lines[0][:120]
                narrative = narrative or " ".join(lines[1:])[:1500]

        # Enforce lengths
        tw = tagline.split()
        if len(tw) > 15:
            tagline = " ".join(tw[:15])
        if len(tw) < 10:
            tagline = " ".join((tw + [""] * (10 - len(tw))))[:120].strip()

        nw = narrative.split()
        if len(nw) > 150:
            narrative = " ".join(nw[:150])
        elif len(nw) < 100:
            narrative = " ".join(nw + [""] * (100 - len(nw))).strip()

        return {"success": True, "tagline": tagline, "narrative": narrative, "model": model}
    except Exception as e:
        return {"success": False, "error": f"HF Inference error: {e}"}
