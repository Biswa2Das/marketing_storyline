import os
import streamlit as st
from marketing_generator_groq import generate_storyline
from description_groq import SceneGenerator, build_default_schema

st.set_page_config(page_title="Marketing Prompt → Scene Builder", page_icon="🎬", layout="wide")

st.title("Marketing Prompt → Storyline → Scene Description")  # Groq-backed
st.caption("Uses Groq Chat Completions (OpenAI-compatible) with Llama-3.3 and structured JSON output.")  # Groq

def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except FileNotFoundError:
        return default

GROQ_API_KEY = _get_secret("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.warning("Missing GROQ_API_KEY in secrets or environment. Set .streamlit/secrets.toml or export GROQ_API_KEY.", icon="⚠️")  # [web:72][web:21]

# Session state
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "storyline" not in st.session_state:
    st.session_state.storyline = None
if "scenes" not in st.session_state:
    st.session_state.scenes = None
if "error" not in st.session_state:
    st.session_state.error = None

with st.sidebar:
    st.subheader("Controls")
    num_scenes = st.slider("Number of scenes", 2, 8, 4, 1)  # UI unchanged
    video_length = st.selectbox("Video length", ["10-second", "15-second", "20-second", "30-second"], index=2)  # UI unchanged
    creativity = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)  # UI unchanged
    st.divider()
    if st.button("Clear All", use_container_width=True):
        st.session_state.prompt = ""
        st.session_state.storyline = None
        st.session_state.scenes = None
        st.session_state.error = None
        st.rerun()

st.session_state.prompt = st.text_area(
    "Marketing prompt",
    value=st.session_state.prompt or "",
    height=160,
    placeholder="Eco-friendly water bottle for Gen Z; emphasize sustainability, UGC hooks, and social proof.",
)

run = st.button("Generate Storyline → Scenes", type="primary", use_container_width=True)

if run:
    st.session_state.error = None
    prompt = (st.session_state.prompt or "").strip()
    if not prompt:
        st.error("Please enter a prompt.")
    elif not GROQ_API_KEY:
        st.error("GROQ_API_KEY missing. Add it to Streamlit secrets or environment.")
    else:
        # Step 1: Storyline
        with st.spinner("Creating storyline..."):
            story_res = generate_storyline(prompt, groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
        if not story_res.get("success"):
            st.session_state.storyline = None
            st.session_state.scenes = None
            st.session_state.error = story_res.get("error", "Storyline generation failed.")
        else:
            storyline = {
                "tagline": story_res.get("tagline", ""),
                "narrative": story_res.get("narrative", ""),
                "model": story_res.get("model", ""),
            }
            st.session_state.storyline = storyline

            # Step 2: Scenes via structured JSON
            with st.spinner("Creating scene descriptions..."):
                sg = SceneGenerator(
                    groq_api_key=GROQ_API_KEY,
                    model="llama-3.3-70b-versatile",
                    temperature=creativity,
                )
                schema = build_default_schema()
                scenes = sg.generate_scenes(
                    storyline=storyline,
                    num_scenes=num_scenes,
                    video_length=video_length,
                    json_schema=schema,
                )
            if not scenes:
                st.session_state.scenes = None
                st.session_state.error = "Scene generation failed."
            else:
                st.session_state.scenes = scenes

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.storyline:
    with st.expander("Storyline", expanded=True):
        st.text_area("Tagline", value=st.session_state.storyline.get("tagline", ""), height=80, disabled=True)
        st.text_area("Narrative", value=st.session_state.storyline.get("narrative", ""), height=180, disabled=True)

if st.session_state.scenes:
    st.subheader("Scene Descriptions")
    st.json({"scenes": st.session_state.scenes})

