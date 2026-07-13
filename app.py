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
# Per-emotion accent colors (vivid, used on white bg)
# -----------------------------
EMOTION_COLORS = {
    "Angry":    ("#FF5F6D", "#C62828"),
    "Disgust":  ("#66BB6A", "#2E7D32"),
    "Fear":     ("#AB47BC", "#6A1B9A"),
    "Happy":    ("#FFC107", "#FB8C00"),
    "Sad":      ("#42A5F5", "#1565C0"),
    "Surprise": ("#EC407A", "#AD1457"),
    "Neutral":  ("#90A4AE", "#546E7A"),
}
DEFAULT_COLOR = ("#7C83FD", "#5C63C4")

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
# Custom CSS — clean white / SaaS-dashboard style
# -----------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: #F6F7FB;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1150px;
    }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #EDEFF5;
    }

    .hero-title {
        text-align: center;
        font-size: 2.7rem;
        font-weight: 900;
        letter-spacing: 0.3px;
        color: #1F2544;
        margin-bottom: 0.1rem;
    }
    .hero-title span {
        background: linear-gradient(90deg, #7C83FD, #22C1A0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        text-align: center;
        color: #6B7189;
        font-size: 1.02rem;
        margin-bottom: 2rem;
    }

    .white-card {
        background: #FFFFFF;
        border: 1px solid #EDEFF5;
        border-radius: 18px;
        padding: 1.5rem 1.7rem;
        height: 100%;
        box-shadow: 0 4px 20px rgba(31, 37, 68, 0.06);
    }

    .card-heading {
        color: #6B7189;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }

    .sidebar-box {
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-box .label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        opacity: 0.75;
        margin-bottom: 0.15rem;
    }
    .sidebar-box .value {
        font-size: 1.02rem;
        font-weight: 700;
    }

    .box-purple { background: #F1EEFF; }
    .box-purple .label, .box-purple .value { color: #6C4FD9; }

    .box-blue { background: #E7F3FF; }
    .box-blue .label, .box-blue .value { color: #1877C9; }

    .box-green { background: #E7FBEF; }
    .box-green .label, .box-green .value { color: #1E9E5A; }

    .box-peach { background: #FFF1E3; }
    .box-peach .label, .box-peach .value { color: #D9772E; }

    .note-box {
        background: #FDF1F7;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        color: #B3467C;
        font-size: 0.82rem;
        line-height: 1.4;
    }

    .result-emoji-wrap {
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .result-emoji {
        font-size: 4.3rem;
    }

    .result-label {
        text-align: center;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: 0.6px;
        margin-top: 0.15rem;
        margin-bottom: 1.4rem;
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
        color: #444B6E;
        font-weight: 600;
    }
    .score-pct {
        width: 40px;
        font-size: 0.8rem;
        color: #8B93B8;
        text-align: right;
        font-weight: 700;
    }
    .bar-track {
        flex: 1;
        background: #EEF0F8;
        border-radius: 8px;
        height: 11px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }

    .placeholder-box {
        border: 1.5px dashed #D8DCEB;
        border-radius: 18px;
        padding: 3.2rem 1rem;
        text-align: center;
        color: #9BA1BC;
        background: #FBFBFE;
    }

    .crop-caption {
        text-align: center;
        color: #8B93B8;
        font-size: 0.78rem;
        margin-top: 0.4rem;
    }

    .warn-banner {
        background: #FFF6E5;
        border: 1px solid #FFE3A3;
        color: #A9760C;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    .ok-banner {
        background: #E9FBF1;
        border: 1px solid #B9EFD1;
        color: #1E9E5A;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1.5px dashed #D8DCEB;
        border-radius: 16px;
        padding: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="hero-title">✨ <span>Facial Emotion Recognition AI</span></p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Powered by Machine Learning &middot; HOG Features + Linear SVM</p>', unsafe_allow_html=True)

# -----------------------------
# Sidebar — each box a distinct pastel shade
# -----------------------------
with st.sidebar:
    st.markdown("## 📊 Model Information")

    st.markdown(
        '<div class="sidebar-box box-purple"><div class="label">Algorithm</div>'
        '<div class="value">Linear SVM</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-box box-blue"><div class="label">Feature Extraction</div>'
        '<div class="value">HOG (Histogram of Oriented Gradients)</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-box box-green"><div class="label">Dataset</div>'
        '<div class="value">FER2013</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-box box-peach"><div class="label">Accuracy</div>'
        '<div class="value">~43% (7-class)</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<div class="note-box">FER2013 is a genuinely hard benchmark — even '
        'human raters agree on labels only ~65-70% of the time. This classical '
        'HOG + linear SVM pipeline is a lightweight baseline, not a CNN, so '
        'treat results as indicative rather than exact.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------
# Upload + Layout
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a face image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

col_img, col_result = st.columns([1, 1], gap="large")
result = None

if uploaded_file is not None:
    # --- Step 1: read + display the uploaded image ---
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with col_img:
            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            st.markdown('<p class="card-heading">Uploaded Image</p>', unsafe_allow_html=True)
            safe_image(pil_image)
            st.markdown('</div>', unsafe_allow_html=True)

        read_ok = True
    except Exception as e:
        read_ok = False
        with col_img:
            st.error("Couldn't read this image file. Please try a different one.")
            with st.expander("Technical details"):
                st.code(str(e))

    # --- Step 2: run prediction (isolated so it can't take down the page) ---
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

                st.markdown('<div class="white-card">', unsafe_allow_html=True)
                st.markdown('<p class="card-heading">Prediction</p>', unsafe_allow_html=True)

                if not result["face_found"]:
                    st.markdown(
                        '<div class="warn-banner">⚠️ No face clearly detected — '
                        'scored the full image. For best accuracy, upload a '
                        'photo where the face is clearly visible and unobstructed.</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("🛠️ Why wasn't a face detected? (debug info)"):
                        st.json(result.get("face_debug", {}))
                else:
                    st.markdown(
                        '<div class="ok-banner">✅ Face detected and cropped '
                        'automatically before scoring.</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="result-emoji-wrap"><span class="result-emoji">{result["emoji"]}</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="result-label" style="color:{c2};">{result["emotion"].upper()}</p>',
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

        # --- Step 3: optional debug preview, fully isolated ---
        if result and result.get("face_crop_bgr") is not None:
            try:
                crop = result["face_crop_bgr"]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    with st.expander("🔍 See exactly what the model scored (the cropped input)"):
                        st.image(crop_rgb, width=200)
                        st.markdown(
                            '<p class="crop-caption">This is the crop — not the full '
                            'photo — that was fed into the model.</p>',
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass  # purely cosmetic; never let this break the page

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
