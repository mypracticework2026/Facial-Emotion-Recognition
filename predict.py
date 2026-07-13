import cv2
import numpy as np
import joblib
from skimage.feature import hog


# Load trained model
model = joblib.load("emotion_model.pkl")


# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


def extract_hog_features(image):

    # Convert PIL image to numpy array
    image = np.array(image)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize image
    resized = cv2.resize(gray, (48, 48))


    # Extract HOG features
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )


    return features



def predict_emotion(image):

    # Extract features
    features = extract_hog_features(image)


    # Reshape because model expects 2D input
    features = features.reshape(1, -1)


    # Make prediction
    prediction = model.predict(features)


    # Convert number to emotion name
    emotion = emotion_labels[prediction[0]]


    return emotion
