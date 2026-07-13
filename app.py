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
# Safe image display helper
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
# Custom CSS — dark, technical aesthetic
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

    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    .stApp > header {
        background: #0A0A0D !important;
        background-image: none !important;
    }

    div[data-testid="stDecoration"] {
        background: linear-gradient(90deg, #E8B84B, #2DD4BF) !important;
        height: 2px !important;
    }

    .block-container {
        padding-top: 2.2rem;
        max-width: 1200px;
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

    /* ---------- Cards ---------- */
    .panel {
        background: #131318;
        border: 1px solid #22222B;
        border-radius: 12px;
        padding: 1.5rem 1.7rem;
        height: 100%;
        position: relative;
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

    .placeholder-box {
        border: 1px dashed #26262F;
        border-radius: 12px;
        padding: 3.4rem 1rem;
        text-align: center;
        color: #565A6E;
        background: #0F0F13;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Eyebrow (unchanged)
# -----------------------------
st.markdown(
    '<div class="eyebrow"><span class="dot"></span> COMPUTER VISION · BIOMETRIC ANALYSIS ENGINE</div>',
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
# Subtitle (unchanged)
# -----------------------------
st.markdown(
    '<p class="hero-subtitle">Upload a photo and the engine locates the face, '
    'extracts HOG gradient features, and classifies the expression across '
    'seven emotional states in real time.</p>',
    unsafe_allow_html=True,
)

# -----------------------------
# Animated Emotion Strip (unchanged)
# -----------------------------
st.markdown(
    '''
    <div class="emo-strip">
        <div class="emo-card" style="--accent:#FF6B5B;">
            <span class="emo-emoji">😠</span>
            <span class="emo-label">Angry</span>
        </div>
        <div class="emo-card" style="--accent:#7FD858;">
            <span class="emo-emoji">🤢</span>
            <span class="emo-label">Disgust</span>
        </div>
        <div class="emo-card" style="--accent:#B78CFF;">
            <span class="emo-emoji">😨</span>
            <span class="emo-label">Fear</span>
        </div>
        <div class="emo-card emo-featured" style="--accent:#FFD166;">
            <div class="scan-frame">
                <span class="corner tl"></span>
                <span class="corner tr"></span>
                <span class="corner bl"></span>
                <span class="corner br"></span>
                <span class="landmark" style="top:22%; left:30%;"></span>
                <span class="landmark" style="top:22%; left:70%;"></span>
                <span class="landmark" style="top:55%; left:50%;"></span>
                <span class="landmark" style="top:78%; left:35%;"></span>
                <span class="landmark" style="top:78%; left:65%;"></span>
                <span class="emo-emoji">😊</span>
            </div>
            <span class="emo-label">Happy</span>
        </div>
        <div class="emo-card" style="--accent:#5EC8E8;">
            <span class="emo-emoji">😢</span>
            <span class="emo-label">Sad</span>
        </div>
        <div class="emo-card" style="--accent:#FF7FC0;">
            <span class="emo-emoji">😲</span>
            <span class="emo-label">Surprise</span>
        </div>
        <div class="emo-card" style="--accent:#9AA1B5;">
            <span class="emo-emoji">😐</span>
            <span class="emo-label">Neutral</span>
        </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar — spec sheet style (unchanged)
# -----------------------------
with st.sidebar:
    st.markdown('<p class="spec-title">◆ Model Specification</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="spec-row" style="--accent:#E8B84B;"><div class="label">Algorithm</div>'
        '<div class="value">Linear SVM</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="spec-row" style="--accent:#2DD4BF;"><div class="label">Feature Extraction</div>'
        '<div class="value">HOG (Histogram of Oriented Gradients)</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="spec-row" style="--accent:#B78CFF;"><div class="label">Dataset</div>'
        '<div class="value">FER2013</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="spec-row" style="--accent:#9AA1B5;"><div class="label">Accuracy</div>'
        '<div class="value">~43% · 7-class</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="note-box">FER2013 is a genuinely hard benchmark — even '
        'human raters agree on labels only ~65-70% of the time. This classical '
        'HOG + linear SVM pipeline is a lightweight baseline, not a CNN, so '
        'treat results as indicative rather than exact.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------
# Upload + Layout (unchanged)
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a face image — JPG or PNG",
    type=["jpg", "jpeg", "png"],
)

col_img, col_result = st.columns([1, 1], gap="large")
result = None

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with col_img:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<p class="card-heading">Input Image</p>', unsafe_allow_html=True)
            safe_image(pil_image)
            st.markdown('</div>', unsafe_allow_html=True)

        read_ok = True
    except Exception as e:
        read_ok = False
        with col_img:
            st.error("Couldn't read this image file. Please try a different one.")
            with st.expander("Technical details"):
                st.code(str(e))

    if read_ok:
        with col_result:
            with st.spinner("Analyzing facial expression..."):
                try:
                    result = predict_emotion(image_bgr)
                except Exception as e:
                    result = None
                    st.error(
                        "Couldn't analyze this image. Try a clearer, "
                        "front-facing photo with the face visible."
                    )
                    with st.expander("Technical details"):
                        st.code(str(e))

            if result:
                c1, c2 = EMOTION_COLORS.get(result["emotion"], DEFAULT_COLOR)

                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<p class="card-heading">Analysis Result</p>', unsafe_allow_html=True)

                if not result["face_found"]:
                    st.markdown(
                        '<div class="warn-banner">⚠ No face clearly detected — '
                        'scored the full image.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="ok-banner">✓ Face detected and cropped.</div>',
                        unsafe_allow_html=True,
                    )

                if result["scores"]:
                    top_score = max(result["scores"].values())
                    if top_score < 0.30:
                        st.markdown(
                            f'<div class="warn-banner">◐ Low confidence ({round(top_score * 100)}%) — '
                            'treat the result as a rough guess.</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    f'<div class="result-emoji-wrap"><span class="result-emoji">{result["emoji"]}</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="result-label" style="color:{c1};">{result["emotion"].upper()}</p>',
                    unsafe_allow_html=True,
                )

                if result["scores"]:
                    sorted_scores = sorted(
                        result["scores"].items(), key=lambda x: x[1], reverse=True
                    )
                    for label, score in sorted_scores:
                        pct = max(0, min(100, round(float(score) * 100)))
                        bc1, bc2 = EMOTION_COLORS.get(label, DEFAULT_COLOR)
                        st.markdown(
                            f'''
                            <div class="score-row">
                                <div class="score-label">{label}</div>
                                <div class="bar-track">
                                    <div class="bar-fill" style="width:{pct}%; background:linear-gradient(90deg, {bc2}, {bc1});"></div>
                                </div>
                                <div class="score-pct">{pct}%</div>
                            </div>
                            ''',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("Confidence breakdown unavailable.")

                st
