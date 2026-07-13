import streamlit as st
import cv2
import numpy as np
from PIL import Image
from predict import predict_emotion

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Per-emotion accent colors (glossy gradients)
# -----------------------------
EMOTION_COLORS = {
    "Angry":    ("#FF5F6D", "#B71C1C"),
    "Disgust":  ("#8BC34A", "#33691E"),
    "Fear":     ("#9C27B0", "#4A148C"),
    "Happy":    ("#FFD200", "#F7971E"),
    "Sad":      ("#4FACFE", "#00567D"),
    "Surprise": ("#F953C6", "#B91D73"),
    "Neutral":  ("#B0BEC5", "#546E7A"),
}
DEFAULT_COLOR = ("#8E9AAF", "#4A5568")

# -----------------------------
# Custom CSS — glassmorphism + gradients
# -----------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at 15% 10%, #1c1440 0%, #0a0e27 45%, #05060f 100%);
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1150px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14102b 0%, #0a0e27 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .hero-title {
        text-align: center;
        font-size: 2.9rem;
        font-weight: 900;
        letter-spacing: 0.5px;
        background: linear-gradient(90deg, #7F5AF0, #2CB67D, #FFD200);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        animation: shine 6s linear infinite;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }

    .hero-subtitle {
        text-align: center;
        color: #9CA3C2;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        letter-spacing: 0.3px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        height: 100%;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }

    .card-heading {
        color: #C9CEE8;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
        opacity: 0.75;
    }

    .result-emoji-wrap {
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .result-emoji {
        font-size: 4.5rem;
        filter: drop-shadow(0 0 25px var(--glow-color, rgba(124,92,240,0.55)));
    }

    .result-label {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, var(--c1, #7F5AF0), var(--c2, #2CB67D));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .score-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .score-label {
        width: 78px;
        font-size: 0.85rem;
        color: #C9CEE8;
        font-weight: 500;
    }
    .score-pct {
        width: 40px;
        font-size: 0.8rem;
        color: #8B93B8;
        text-align: right;
        font-weight: 600;
    }
    .bar-track {
        flex: 1;
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 11px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
        box-shadow: 0 0 10px rgba(0,0,0,0.3) inset;
    }

    .placeholder-box {
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 3.2rem 1rem;
        text-align: center;
        color: #6B7394;
        background: rgba(255,255,255,0.02);
    }

    .crop-caption {
        text-align: center;
        color: #8B93B8;
        font-size: 0.78rem;
        margin-top: 0.4rem;
    }

    .warn-banner {
        background: rgba(255, 190, 60, 0.12);
        border: 1px solid rgba(255, 190, 60, 0.35);
        color: #FFD98A;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    .ok-banner {
        background: rgba(44, 182, 125, 0.12);
        border: 1px solid rgba(44, 182, 125, 0.35);
        color: #8FE3BE;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 0.6rem;
    }

    [data-testid="stSidebar"] .stAlert {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="hero-title">✨ Facial Emotion Recognition AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Powered by Machine Learning &middot; HOG Features + Linear SVM</p>', unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.success("**Algorithm**\n\nLinear SVM")
    st.success("**Feature Extraction**\n\nHOG (Histogram of Oriented Gradients)")
    st.success("**Dataset**\n\nFER2013")
    st.success("**Accuracy**\n\n~43% (7-class)")
    st.markdown("---")
    st.info("Built with Python, OpenCV, scikit-image, scikit-learn and Streamlit.")
    st.caption(
        "FER2013 is a genuinely hard benchmark — even human raters agree "
        "on labels only ~65-70% of the time. This classical HOG + linear "
        "SVM pipeline is a lightweight baseline, not a CNN, so treat "
        "results as indicative rather than exact."
    )

# -----------------------------
# Upload + Layout
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a face image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

col_img, col_result = st.columns([1, 1], gap="large")

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="card-heading">Uploaded Image</p>', unsafe_allow_html=True)
            st.image(pil_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

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

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<p class="card-heading">Prediction</p>', unsafe_allow_html=True)

                if not result["face_found"]:
                    st.markdown(
                        '<div class="warn-banner">⚠️ No face clearly detected — '
                        'scored the full image. For best accuracy, upload a '
                        'photo where the face is clearly visible and unobstructed.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="ok-banner">✅ Face detected and cropped automatically '
                        'before scoring.</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="result-emoji-wrap" style="--glow-color:{c1}66;">'
                    f'<span class="result-emoji">{result["emoji"]}</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="result-label" style="--c1:{c1};--c2:{c2};">{result["emotion"].upper()}</p>',
                    unsafe_allow_html=True,
                )

                if result["scores"]:
                    sorted_scores = sorted(
                        result["scores"].items(), key=lambda x: x[1], reverse=True
                    )
                    for label, score in sorted_scores:
                        pct = max(0, min(100, round(score * 100)))
                        bc1, bc2 = EMOTION_COLORS.get(label, DEFAULT_COLOR)
                        st.markdown(
                            f'''
                            <div class="score-row">
                                <div class="score-label">{label}</div>
                                <div class="bar-track">
                                    <div class="bar-fill" style="width:{pct}%; background:linear-gradient(90deg, {bc1}, {bc2});"></div>
                                </div>
                                <div class="score-pct">{pct}%</div>
                            </div>
                            ''',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("Confidence breakdown unavailable for this model type.")

                st.markdown('</div>', unsafe_allow_html=True)

        # Show what the model actually scored, for transparency/debugging
        if result and result.get("face_crop_bgr") is not None:
            crop_rgb = cv2.cvtColor(result["face_crop_bgr"], cv2.COLOR_BGR2RGB)
            with st.expander("🔍 See exactly what the model scored (the cropped input)"):
                st.image(crop_rgb, width=200)
                st.markdown(
                    '<p class="crop-caption">This 48×48 grayscale crop — not the full '
                    'photo — is what was fed into the model.</p>',
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error("Something went wrong while reading this image. Please try a different file.")
        with st.expander("Technical details"):
            st.code(str(e))

else:
    with col_img:
        st.markdown(
            '<div class="placeholder-box">🖼️<br><br>Your uploaded photo will appear here</div>',
            unsafe_allow_html=True,
        )
    with col_result:
        st.markdown(
            '<div class="placeholder-box">📈<br><br>Prediction and confidence scores will appear here</div>',
            unsafe_allow_html=True,
        )
