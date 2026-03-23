import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    cnn_model = load_model("cnn_model.h5")
    mobilenet_model = load_model("mobilenet_model.h5")

    # Dummy forward pass to initialize Sequential models
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    cnn_model.predict(dummy_input)
    mobilenet_model.predict(dummy_input)

    return cnn_model, mobilenet_model

cnn_model, mobilenet_model = load_models()

# ----------------------------
# Class Labels
# ----------------------------
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_image(img: Image.Image, target_size=(224,224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Prediction
# ----------------------------
def predict(model, img: Image.Image):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return classes[class_idx], round(confidence*100, 2)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Waste Classifier", layout="centered")
st.title("Waste Image Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_option = st.selectbox("Select Model", ["CNN", "MobileNet"])

if uploaded_file and st.button("Predict"):
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=400)

        model = cnn_model if model_option == "CNN" else mobilenet_model
        predicted_class, confidence = predict(model, img)

        st.success(f"Predicted Class: **{predicted_class}**")
        st.info(f"Confidence: **{confidence}%**")

    except Exception as e:
        st.error(f"Error: {str(e)}")