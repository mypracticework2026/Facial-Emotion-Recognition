import cv2
import joblib
import numpy as np
from skimage.feature import hog

MODEL_PATH = "emotion_model.pkl"
model = joblib.load(MODEL_PATH)

# Haar cascade ships with opencv-python / opencv-python-headless — no
# extra dependency needed.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Standard FER2013 label order — this is the order the original Kaggle
# CSV encodes emotions in if your model was trained on the raw integer
# 'emotion' column (0-6). If your model was trained on string labels
# instead, this file handles that automatically too.
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOJI_MAP = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐",
}


def detect_and_crop_face(image_bgr):
    """
    Finds the largest face in the image and crops to it (with a small
    margin) so the model sees roughly what FER2013 was trained on:
    a tight, face-only crop rather than a full photo with background,
    shoulders, etc.

    Returns (cropped_bgr, face_found: bool). If no face is detected,
    returns the original image unchanged and face_found=False so the
    UI can warn the user that accuracy may suffer.
    """
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return image_bgr, False

    # Use the largest detected face (most likely the main subject)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    margin = int(0.2 * w)
    h_img, w_img = image_bgr.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w_img, x + w + margin)
    y1 = min(h_img, y + h + margin)

    cropped = image_bgr[y0:y1, x0:x1]
    return cropped, True


def _extract_features(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = cv2.equalizeHist(gray)  # improves contrast on real-world photos

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )
    return np.array(features).reshape(1, -1)


def _label_to_name(raw_label):
    """Normalize whatever the model outputs (int class index OR string
    label) into one of our canonical emotion names."""
    if isinstance(raw_label, (str, np.str_)):
        name = str(raw_label).strip().lower()
        for lbl in EMOTION_LABELS:
            if lbl.lower() == name:
                return lbl
        return name.capitalize()

    idx = int(raw_label)
    if 0 <= idx < len(EMOTION_LABELS):
        return EMOTION_LABELS[idx]
    return f"Class {idx}"


def _get_confidence_scores(features):
    """Best-effort per-emotion confidence. Returns None if the model
    exposes neither predict_proba nor decision_function."""
    class_labels = [_label_to_name(c) for c in model.classes_]

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(features)[0]
            return dict(zip(class_labels, probs))
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            scores = np.atleast_1d(model.decision_function(features)[0])
            exp = np.exp(scores - np.max(scores))
            probs = exp / exp.sum()
            return dict(zip(class_labels, probs))
        except Exception:
            pass

    return None


def predict_emotion(image_bgr):
    """
    Takes a BGR numpy image, returns:
        {
            "emotion": "Happy",
            "emoji": "😊",
            "scores": {"Angry": 0.03, "Happy": 0.61, ...} or None,
            "face_crop_bgr": <the exact crop the model scored>,
            "face_found": True/False,
        }
    """
    face_crop, face_found = detect_and_crop_face(image_bgr)

    features = _extract_features(face_crop)
    raw_pred = model.predict(features)[0]
    emotion = _label_to_name(raw_pred)
    scores = _get_confidence_scores(features)

    return {
        "emotion": emotion,
        "emoji": EMOJI_MAP.get(emotion, "🙂"),
        "scores": scores,
        "face_crop_bgr": face_crop,
        "face_found": face_found,
    }
