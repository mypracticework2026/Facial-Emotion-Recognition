import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog

MODEL_PATH = "emotion_model.pkl"
model = joblib.load(MODEL_PATH)


def _load_face_cascades():
    """
    Loads every Haar cascade variant we can find (default + alt + alt2).
    Different variants catch different faces — trying several drastically
    improves real-world hit rate compared to relying on just one.
    Returns a list of successfully-loaded CascadeClassifier objects
    (possibly empty, never raises).
    """
    filenames = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt.xml",
    ]

    base_dirs = []
    try:
        base_dirs.append(cv2.data.haarcascades)
    except AttributeError:
        pass
    try:
        base_dirs.append(os.path.join(os.path.dirname(cv2.__file__), "data"))
    except Exception:
        pass

    cascades = []
    for filename in filenames:
        for base in base_dirs:
            path = os.path.join(base, filename)
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    cascades.append(cascade)
                break  # found this variant, move to next filename
    return cascades


FACE_CASCADES = _load_face_cascades()

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

    Tries every loaded cascade variant at a couple of sensitivity
    levels before giving up, since a single cascade + single parameter
    set misses a surprising number of real-world photos.

    Returns (cropped_bgr, face_found: bool, debug: dict).
    """
    debug = {
        "cascades_loaded": len(FACE_CASCADES),
        "image_shape": image_bgr.shape,
        "attempts": [],
    }

    if not FACE_CASCADES:
        debug["reason"] = "No Haar cascades could be loaded on this environment."
        return image_bgr, False, debug

    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray_full)

    # (image used, minNeighbors, minSize) — from strict to lenient
    param_sets = [
        (gray_eq, 5, (60, 60)),
        (gray_eq, 3, (40, 40)),
        (gray_full, 3, (40, 40)),
        (gray_eq, 2, (30, 30)),
    ]

    best_face = None
    for cascade in FACE_CASCADES:
        for img, min_neighbors, min_size in param_sets:
            faces = cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=min_size
            )
            debug["attempts"].append(
                {"min_neighbors": min_neighbors, "min_size": min_size, "found": len(faces)}
            )
            if len(faces) > 0:
                candidate = max(faces, key=lambda f: f[2] * f[3])
                if best_face is None or candidate[2] * candidate[3] > best_face[2] * best_face[3]:
                    best_face = candidate
        if best_face is not None:
            break  # this cascade found something, no need to try the rest

    if best_face is None:
        debug["reason"] = "No face matched by any cascade/parameter combination."
        return image_bgr, False, debug

    x, y, w, h = best_face
    margin = int(0.25 * w)
    h_img, w_img = image_bgr.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w_img, x + w + margin)
    y1 = min(h_img, y + h + margin)

    cropped = image_bgr[y0:y1, x0:x1]
    debug["chosen_box"] = [int(x), int(y), int(w), int(h)]
    return cropped, True, debug


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
            "face_debug": {...diagnostic info...},
        }
    """
    face_crop, face_found, face_debug = detect_and_crop_face(image_bgr)

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
        "face_debug": face_debug,
    }
