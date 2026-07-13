### 🔥 The old title line:
```python
st.markdown('<p class="hero-title">Facial Emotion Detector</p>', unsafe_allow_html=True)
```

is replaced with your **new colorful rectangle heading**.

Everything else — CSS, sidebar, uploader, animations, layout — is **100% identical** to your original file.

---

# ⭐ **FULL UPDATED `app.py` — COPY & PASTE**

```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from predict import predict_emotion

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Facial Emotion Detector",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Per-emotion accent colors — jewel tones tuned for a dark surface
# -----------------------------
EMOTION_COLORS = {
    "Angry":    ("#FF6B5B", "#C4362A"),
    "Disgust":  ("#7FD858", "#3F8F24"),
    "Fear":     ("#B78CFF", "#7C4FDB"),
    "Happy":    ("#FFD166", "#E8B84B"),
    "Sad":      ("#5EC8E8", "#2A8FC4"),
    "Surprise": ("#FF7FC0", "#D6428F"),
    "Neutral":  ("#9AA1B5", "#5C6072"),
}
DEFAULT_COLOR = ("#E8B84B", "#B8903A")

# -----------------------------
# Safe image display helper (works across Streamlit versions)
# -----------------------------
def safe_image(img, **kwargs):
    try:
        st.image(img, use_container_width=True, **kwargs)
    except TypeError:
        try:
            st.image(img, use_column_width=True, **kwargs)
        except TypeError:
            st.image(img, **kwargs)

# -----------------------------
# Custom CSS — dark, technical, "biometric analysis console" aesthetic
# -----------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    html, body {
        background: #0A0A0D !important;
    }

    .stApp {
        background: #0A0A0D;
        background-image:
            radial-gradient(circle at 8% 0%, rgba(232, 184, 75, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 95% 15%, rgba(45, 212, 191, 0.05) 0%, transparent 40%);
    }

    /* Kill every flavor of Streamlit's default white header/toolbar chrome */
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    .stApp > header {
        background: #0A0A0D !important;
        background-image: none !important;
    }

    header[data-testid="stHeader"] * {
        background: transparent !important;
    }

    header[data-testid="stHeader"] svg {
        fill: #C4C7D4 !important;
    }

    div[data-testid="stDecoration"] {
        background: linear-gradient(90deg, #E8B84B, #2DD4BF) !important;
        height: 2px !important;
    }

    .block-container {
        padding-top: 2.2rem;
        max-width: 1200px;
    }

    section[data-testid="stSidebar"] {
        background: #0E0E12;
        border-right: 1px solid #1E1E26;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }

    /* ---------- Hero ---------- */
    .eyebrow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #E8B84B;
        margin-bottom: 1rem;
    }
    .eyebrow .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #E8B84B;
        box-shadow: 0 0 8px 2px rgba(232, 184, 75, 0.6);
    }

    .hero-title {
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4.4rem;
        font-weight: 700;
        letter-spacing: -1.5px;
        color: #F3F4F8;
        margin-bottom: 0.6rem;
        line-height: 1.05;
    }
    @media (max-width: 640px) {
        .hero-title { font-size: 2.6rem; }
    }

    .hero-subtitle {
        text-align: center;
        color: #8B90A3;
        font-size: 1.05rem;
        max-width: 560px;
        margin: 0 auto 1.6rem auto;
        line-height: 1.5;
    }

    .scan-line {
        width: 100%;
        max-width: 340px;
        height: 2px;
        margin: 0 auto 2rem auto;
        background: linear-gradient(90deg, transparent, #E8B84B, #2DD4BF, transparent);
        background-size: 200% 100%;
        animation: scan 5s linear infinite;
        border-radius: 2px;
    }
    @keyframes scan {
        to { background-position: -200% 0; }
    }

    /* ---------- Animated emotion face-strip ---------- */
    .emo-strip {
        display: flex;
        justify-content: center;
        align-items: flex-end;
        gap: 14px;
        flex-wrap: wrap;
        margin: 0.5rem auto 2.4rem auto;
        max-width: 760px;
    }

    .emo-card {
        background: #131318;
        border: 1px solid #22222B;
        border-radius: 14px;
        padding: 1rem 0.7rem 0.85rem 0.7rem;
        width: 84px;
        text-align: center;
        cursor: default;
        transition: transform 0.35s cubic-bezier(.34,1.56,.64,1), box-shadow 0.35s ease, border-color 0.35s ease;
    }

    .emo-card:hover {
        transform: translateY(-8px) scale(1.1);
        border-color: var(--accent, #E8B84B);
        box-shadow: 0 10px 26px -8px var(--accent, #E8B84B);
        z-index: 2;
    }

    .emo-emoji {
        font-size: 2rem;
        display: block;
        transition: transform 0.35s ease;
    }

    .emo-card:hover .emo-emoji {
        animation: wiggle 0.55s ease;
    }

    @keyframes wiggle {
        0%, 100% { transform: rotate(0deg) scale(1); }
        25%      { transform: rotate(-14deg) scale(1.15); }
        75%      { transform: rotate(14deg) scale(1.15); }
    }

    .emo-label {
        display: block;
        margin-top: 0.55rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        font-weight: 500;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        color: #6C7189;
    }

    /* Spotlighted card — the "actively scanned" face */
    .emo-featured {
        width: 108px;
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent) inset, 0 0 28px -10px var(--accent);
        animation: float 3.2s ease-in-out infinite;
    }
    .emo-featured .emo-emoji {
        font-size: 2.6rem;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-6px); }
    }

    .scan-frame {
        position: relative;
        width: 100%;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .scan-frame .corner {
        position: absolute;
        width: 10px;
        height: 10px;
        border: 2px solid var(--accent);
        opacity: 0.9;
        animation: pulse 2s ease-in-out infinite;
    }
    .corner.tl { top: -8px;    left: -8px;   border-right: none;  border-bottom: none; }
    .corner.tr { top: -8px;    right: -8px;  border-left: none;   border-bottom: none; }
    .corner.bl { bottom: -8px; left: -8px;   border-right: none;  border-top: none; }
    .corner.br { bottom: -8px; right: -8px;  border-left: none;   border-top: none; }
    @keyframes pulse {
        0%, 100% { opacity: 0.4; }
        50%      { opacity: 1; }
    }
    .landmark {
        position: absolute;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 5px var(--accent);
    }

    /* ---------- Cards ---------- */
    .panel {
        background: #131318;
        border: 1px solid #22222B;
        border-radius: 12px;
        padding: 1.5rem 1.7rem;
        height: 100%;
        position: relative;
    }
    .panel::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.10), transparent);
    }

    .card-heading {
        color: #6C7189;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* ---------- Sidebar spec sheet ---------- */
    .spec-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #F3F4F8;
        margin-bottom: 1.1rem;
        letter-spacing: 0.2px;
    }
    .spec-row {
        background: #16161C;
        border: 1px solid #212129;
        border-left: 3px solid var(--accent, #E8B84B);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.7rem;
    }
    .spec-row .label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        font-weight: 500;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #6C7189;
        margin-bottom: 0.25rem;
    }
    .spec-row .value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #E4E5EC;
    }

    .note-box {
        background: #16161C;
        border: 1px solid #212129;
        border-radius: 10px;
        padding: 0.9rem 1.05rem;
        color: #8B90A3;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    /* ---------- Result ---------- */
    .result-emoji-wrap {
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .result-emoji {
        font-size: 4.2rem;
    }

    .result-label {
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    .score-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 11px;
    }
    .score-label {
        width: 82px;
        font-size: 0.82rem;
        color: #B0B4C4;
        font-weight: 500;
    }
    .score-pct {
        width: 42px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: #6C7189;
        text-align: right;
        font-weight: 500;
    }
    .bar-track {
        flex: 1;
        background: #1C1C24;
        border-radius: 6px;
        height: 9px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }

    .placeholder-box {
        border: 1px dashed #26262F;
        border-radius: 12px;
        padding: 3.4rem 1rem;
        text-align: center;
        color: #565A6E;
        background: #0F0F13;
        font-size: 0.9rem;
    }

    .crop-caption {
        text-align: center;
        color: #6C7189;
        font-size: 0.76rem;
        margin-top: 0.5rem;
    }

    .warn-banner {
        background: rgba(232, 184, 75, 0.08);
        border: 1px solid rgba(232, 184, 75, 0.28);
        color: #E8B84B;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.84rem;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    .ok-banner {
        background: rgba(45, 212, 191, 0.08);
        border: 1px solid rgba(45, 212, 191, 0.28);
        color: #2DD4BF;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stFileUploader"] {
        background: #0F0F13;
        border: 1px dashed #262630;
        border-radius: 12px;
        padding: 0.6rem;
    }
    div[data-testid="stFileUploader"] section {
        background: transparent;
    }

    /* Streamlit widget text tuning */
    p, span, label { color: #C4C7D4; }
    .stMarkdown, .stCaption { color: #8B90A3; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero
# -----------------------------
st.markdown(
    '<div class="eyebrow"><span class="dot"></span> COMPUTER VISION &middot; BIOMETRIC ANALYSIS ENGINE</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# NEW TITLE BLOCK (Option A)
# -----------------------------
st.markdown(
    """
    <div style="
        display:flex;
        justify-content:center;
        margin-top:5px;
        margin-bottom:10px;
    ">
        <div style="
            padding: 22px 40px;
            border-radius: 18px;
            border: 3px solid;
            border-image: linear-gradient(90deg, #FFD166, #2DD4BF, #B78CFF) 1;
            box-shadow: 0 0 22px rgba(255, 209, 102, 0.25);
            text-align:center;
        ">
            <h1 style="
                font-family: 'Space Grotesk', sans-serif;
                font-size: 5rem;
                font-weight: 700;
                letter-spacing: -1px;
                margin: 0;
                color: #F3F4F8;
            ">
                Facial Emotion Detector
            </h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Subtitle
# -----------------------------
st.markdown(
    '<p class="hero-subtitle">Upload a photo and the engine locates the face, '
    'extracts HOG gradient features, and classifies the expression across '
    'seven emotional states in real time.</p>',
    unsafe_allow_html=True,
)

# -----------------------------
# Animated Emotion Strip
# -----------------------------
st.markdown(
    '''
    <div class="emo-strip">
        <div class="emo-card" style="--accent:#FF6B5B;">
            <span class="emo-emoji">😠</span>
            <span class="emo-label">Angry</span>
        </div>
        <div class="emo-card" style="--accent:#7FD858;">
