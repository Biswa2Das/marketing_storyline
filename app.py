# app.py
import os
import streamlit as st
from marketing_generator import generate_storyline
from description import SceneGenerator, build_default_schema

st.set_page_config(page_title="Marketing Prompt → Scene Builder", page_icon="🎬", layout="wide")

st.title("Marketing Prompt → Storyline → Scene Description")  # [web:154]
st.caption("Uses Hugging Face Inference Providers with Llama-3.1-8B-Instruct and structured JSON output.")  # [web:191]

# Read HF token from Streamlit secrets for cloud; fallback to env for local dev
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.warning("Missing HF_TOKEN in Streamlit secrets. Add it in Deployment → Secrets.", icon="⚠️")  # [web:152]

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
    num_scenes = st.slider("Number of scenes", 2, 8, 4, 1)  # [web:114]
    video_length = st.selectbox("Video length", ["10-second", "15-second", "20-second", "30-second"], index=2)  # [web:114]
    creativity = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)  # [web:114]
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
)  # [web:112]

run = st.button("Generate Storyline → Scenes", type="primary", use_container_width=True)  # [web:114]

if run:
    st.session_state.error = None
    prompt = (st.session_state.prompt or "").strip()
    if not prompt:
        st.error("Please enter a prompt.")
    elif not HF_TOKEN:
        st.error("HF_TOKEN missing. Add it to Streamlit secrets.")
    else:
        # Step 1: Storyline (tagline + narrative)
        with st.spinner("Creating storyline..."):
            story_res = generate_storyline(prompt, hf_token=HF_TOKEN, model="meta-llama/Llama-3.1-8B-Instruct")
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
                    hf_token=HF_TOKEN,
                    model="meta-llama/Llama-3.1-8B-Instruct",
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
        st.text_area("Tagline", value=st.session_state.storyline.get("tagline", ""), height=80, disabled=True)  # [web:112]
        st.text_area("Narrative", value=st.session_state.storyline.get("narrative", ""), height=180, disabled=True)  # [web:112]

if st.session_state.scenes:
    st.subheader("Scene Descriptions")
    st.json({"scenes": st.session_state.scenes})  # [web:110]
