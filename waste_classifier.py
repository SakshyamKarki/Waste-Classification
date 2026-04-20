import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────────
# Load models  (cached so they are loaded only once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():

    cnn_model       = tf.keras.models.load_model("models/cnn_model.h5")
    mobilenet_model = tf.keras.models.load_model("models/mobilenet_model.h5")

    # Warm-up pass so the first real prediction is fast
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    cnn_model.predict(dummy, verbose=0)
    mobilenet_model.predict(dummy, verbose=0)
    return cnn_model, mobilenet_model


cnn_model, mobilenet_model = load_models()

# ──────────────────────────────────────────────────────────────────────────────
# Class labels
# ──────────────────────────────────────────────────────────────────────────────
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(img: Image.Image, target=(224, 224)) -> np.ndarray:
    img = img.convert("RGB").resize(target)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)


def predict(model, img: Image.Image):
    probs     = model.predict(preprocess(img), verbose=0)[0]
    class_idx = int(np.argmax(probs))
    return CLASSES[class_idx], round(float(probs[class_idx]) * 100, 2), probs

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Waste Classifier", layout="centered")
st.title("🗑️ Waste Image Classification")
st.caption("Classify waste into: cardboard · glass · metal · paper · plastic · trash")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_option  = st.selectbox("Select Model", ["CNN", "MobileNetV2 (fine-tuned)"])

if uploaded_file and st.button("Predict"):
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=400)

        model = cnn_model if "CNN" in model_option else mobilenet_model
        predicted_class, confidence, probs = predict(model, img)

        st.success(f"**Predicted Class:** {predicted_class}")
        st.info(f"**Confidence:** {confidence}%")

        # Show probability bar chart for all classes
        st.subheader("Class Probabilities")
        prob_dict = {c: float(p) for c, p in zip(CLASSES, probs)}
        st.bar_chart(prob_dict)

    except Exception as e:
        st.error(f"Error: {e}")