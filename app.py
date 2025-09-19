# app.py
import streamlit as st

# Local modules
from marketing_generator import RobustMarketingGenerator  # class exposes .generate_storyline(...)
from description import SceneGenerator, Config  # class exposes .generate_scenes(...)

st.set_page_config(page_title="Marketing Prompt → Scene Builder", page_icon="🎬", layout="wide")

# ------------- UI Header -------------
st.title("Marketing Prompt → Storyline → Scene Description")  # [web:60]
st.caption("Enter a marketing prompt, generate a storyline, then convert it into structured scene descriptions.")  # [web:60]

# ------------- Session State -------------
if "prompt" not in st.session_state:
    st.session_state.prompt = ""  # [web:23]
if "storyline" not in st.session_state:
    st.session_state.storyline = None  # [web:23]
if "scenes" not in st.session_state:
    st.session_state.scenes = None  # [web:23]
if "error" not in st.session_state:
    st.session_state.error = None  # [web:23]

# ------------- Sidebar -------------
with st.sidebar:
    st.subheader("Controls")  # [web:60]
    num_scenes = st.slider("Number of scenes", min_value=2, max_value=8, value=4, step=1)  # [web:114]
    video_length = st.selectbox("Video length", ["10-second", "15-second", "20-second", "30-second"], index=1)  # [web:114]
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)  # [web:114]
    st.divider()  # [web:60]
    if st.button("Clear All", use_container_width=True):
        st.session_state.prompt = ""  # [web:23]
        st.session_state.storyline = None  # [web:23]
        st.session_state.scenes = None  # [web:23]
        st.session_state.error = None  # [web:23]
        st.rerun()  # [web:105]

# ------------- Prompt Input -------------
st.session_state.prompt = st.text_area(
    "Marketing prompt",
    value=st.session_state.prompt or "",
    height=160,
    placeholder="e.g., Eco-friendly water bottle for Gen Z; emphasize sustainability, UGC hooks, and social proof.",
)  # [web:112]

col_generate, col_dummy = st.columns([1, 3])  # [web:117]
with col_generate:
    run = st.button("Generate Storyline → Scenes", type="primary", use_container_width=True)  # [web:114]

# ------------- Generation Pipeline -------------
if run:
    st.session_state.error = None  # [web:23]
    prompt = (st.session_state.prompt or "").strip()  # [web:23]
    if not prompt:
        st.error("Please enter a prompt.")  # [web:60]
    else:
        # Step 1: Storyline
        with st.spinner("Creating storyline..."):  # [web:60]
            mg = RobustMarketingGenerator()
            res = mg.generate_storyline(prompt)
        if not res.get("success"):
            st.session_state.storyline = None  # [web:23]
            st.session_state.scenes = None  # [web:23]
            st.session_state.error = res.get("error", "Storyline generation failed.")  # [web:23]
        else:
            storyline = {
                "tagline": res.get("tagline", ""),
                "narrative": res.get("narrative", ""),
                "model": res.get("model", ""),
            }
            st.session_state.storyline = storyline  # [web:23]

            # Step 2: Scenes
            with st.spinner("Creating scene descriptions..."):  # [web:60]
                sg = SceneGenerator(Config())
                scenes = sg.generate_scenes(
                    storyline=storyline,
                    num_scenes=num_scenes,
                    video_length=video_length,
                    temperature=temperature,
                )
            if not scenes:
                st.session_state.scenes = None  # [web:23]
                st.session_state.error = "Scene generation failed. Check model logs."  # [web:23]
            else:
                st.session_state.scenes = scenes  # [web:23]

# ------------- Outputs -------------
if st.session_state.error:
    st.error(st.session_state.error)  # [web:60]

# Show storyline (read-only)
if st.session_state.storyline:
    with st.expander("Storyline", expanded=True):
        st.text_area(
            "Tagline",
            value=st.session_state.storyline.get("tagline", ""),
            height=80,
            disabled=True,
        )  # [web:112]
        st.text_area(
            "Narrative",
            value=st.session_state.storyline.get("narrative", ""),
            height=180,
            disabled=True,
        )  # [web:112]

# Show scenes JSON
if st.session_state.scenes:
    st.subheader("Scene Descriptions")  # [web:60]
    st.json({"scenes": st.session_state.scenes}, expanded=2)  # [web:110]

# ------------- Footer -------------
with st.sidebar:
    st.caption("Tip: If imports fail, ensure app.py imports the class RobustMarketingGenerator and SceneGenerator, not a missing free function.")  # [web:86]
