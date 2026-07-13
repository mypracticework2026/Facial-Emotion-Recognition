# 😊 Facial Emotion Recognition AI

A Streamlit web app that predicts human facial emotions from an uploaded photo using classical machine learning — HOG (Histogram of Oriented Gradients) feature extraction paired with a Linear SVM classifier, trained on the FER2013 dataset.

<p align="center">
  <img src="docs/screenshot.png" alt="App screenshot" width="800">
</p>

---

## ✨ Features

- **Automatic face detection & cropping** — uses OpenCV Haar cascades to locate and crop the face before scoring, so predictions match what the model actually learned on instead of scoring an entire photo (background, shoulders, etc.)
- **Per-emotion confidence bars** — see the full probability breakdown across all 7 emotions, not just the top guess
- **Confidence-aware warnings** — flags when face detection failed or when the model's top prediction is low-confidence, so you know when to trust a result
- **Debug panel** — inspect exactly what image crop was fed into the model, and why face detection succeeded or failed on a given photo
- **Clean, responsive UI** — light theme with color-coded emotion indicators

## 🧠 Model Details

| | |
|---|---|
| **Algorithm** | Linear SVM |
| **Feature Extraction** | HOG (Histogram of Oriented Gradients) |
| **Dataset** | [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) |
| **Classes** | Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral |
| **Accuracy** | ~43% (7-class) |

> **Note:** FER2013 is a genuinely difficult benchmark — even human annotators agree on labels only ~65–70% of the time. This is a lightweight classical baseline (HOG + linear SVM), not a deep CNN, so treat predictions as indicative rather than exact. Low-confidence results are flagged in the UI.

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web UI
- [OpenCV](https://opencv.org/) — face detection & image preprocessing
- [scikit-image](https://scikit-image.org/) — HOG feature extraction
- [scikit-learn](https://scikit-learn.org/) — SVM model
- [joblib](https://joblib.readthedocs.io/) — model serialization
- [Pillow](https://python-pillow.org/) / [NumPy](https://numpy.org/) — image handling

## 📂 Project Structure

```
.
├── app.py                  # Streamlit UI
├── predict.py               # Face detection + feature extraction + inference
├── emotion_model.pkl        # Trained Linear SVM model (not included — see below)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your trained model

Place your trained model file as `emotion_model.pkl` in the project root. If the file is larger than 100MB, use [Git LFS](https://git-lfs.com/) to commit it.

### 4. Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (including `emotion_model.pkl`, via Git LFS if needed).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Select this repository and set the main file path to `app.py`.
4. Deploy. First build may take a minute or two while dependencies install.

## 📋 requirements.txt

```
streamlit
opencv-python-headless
scikit-image
scikit-learn
joblib
numpy
pillow
```

> Use `opencv-python-headless` (not `opencv-python`) — the headless build avoids missing system GUI library errors on Streamlit Cloud.

## ⚠️ Limitations

- Classical HOG + SVM accuracy (~43%) is well below what a CNN can achieve on this task — expect the model to struggle with ambiguous expressions, unusual lighting, extreme angles, or faces that don't closely resemble the FER2013 training distribution (e.g. very young, very old, or heavily occluded faces).
- Face detection relies on Haar cascades, which can miss faces in low-resolution, heavily cropped, blurry, or non-frontal photos. When this happens, the app falls back to scoring the full image and flags it in the UI.
- This project is for educational/demonstration purposes and is not intended for clinical, security, or other high-stakes use.

## 📄 License

Add your preferred license here (e.g. MIT).
