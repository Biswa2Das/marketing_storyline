from typing import Dict, Any
from groq import Groq

def _build_prompt(product_input: str) -> str:
    return f"""You are a creative marketing expert. Based on the following product details, create:

1. A short tagline (10–15 words maximum)
2. A full marketing narrative (100–150 words)

Product Details: {product_input}

Format your response exactly as:
TAGLINE: <tagline>
NARRATIVE: <narrative>
"""

def generate_storyline(product_input: str, groq_api_key: str, model: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
    client = Groq(api_key=groq_api_key)  # GROQ_API_KEY from env or secrets

    messages = [
        {"role": "system", "content": "Return only the requested sections."},
        {"role": "user", "content": _build_prompt(product_input)},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )  # OpenAI-compatible chat completions on Groq
        text = resp.choices[0].message.content or ""

        tagline, narrative = "", ""
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("TAGLINE:"):
                tagline = s[len("TAGLINE:"):].strip()
            elif s.startswith("NARRATIVE:"):
                narrative = s[len("NARRATIVE:"):].strip()

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
        return {"success": False, "error": f"Groq chat error: {e}"}



