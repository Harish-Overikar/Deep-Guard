
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
import streamlit as st
from PIL import Image, ImageOps


# ---------------- MODEL CONFIGURATION ----------------

MODEL_PATH = str(Path(__file__).resolve().parent / "deepfake_model.keras")
IMAGE_SIZE = (128, 128)
THRESHOLD = 0.4


# Load the trained model once and cache it
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)


model = load_model()


# ---------------- IMAGE PREPROCESSING ----------------

def preprocess_image(image):
    """Match the exact preprocessing used during training in the notebook."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0

    return tf.expand_dims(image_array, axis=0)


# ---------------- IMAGE PREDICTION ----------------

def predict_image(image):
    image_tensor = preprocess_image(image)
    prediction = model.predict(image_tensor, verbose=0)
    probability = float(np.clip(np.squeeze(prediction), 0.0, 1.0))

    # Class 0 = Real, Class 1 = Fake
    if probability > THRESHOLD:
        label = "Fake"
        confidence = probability * 100
    else:
        label = "Real"
        confidence = (1 - probability) * 100

    print(f"Raw sigmoid output: {probability:.6f}")
    print(f"Predicted class: {label}")
    print(f"Confidence: {confidence:.2f}%")

    return label, confidence