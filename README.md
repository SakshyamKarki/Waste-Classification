Project Title

Smart Waste Image Classifier – CNN & MobileNet with Grad-CAM and Streamlit GUI

Objective

Build an image classification system that classifies waste images into the following categories:

Cardboard
Glass
Metal
Paper
Plastic
Trash

The system uses CNN and MobileNet models, provides confidence scores, and visualizes attention with Grad-CAM heatmaps. A Streamlit GUI allows easy image uploads and predictions.

Dataset Description

Dataset: Kaggle Waste Image Classification

Total classes: 6 (cardboard, glass, metal, paper, plastic, trash)
Images are RGB, varying in size
Training/Validation split: manually split during preprocessing (80% train, 20% validation)
Preprocessing:
Resize images to 224×224
Normalize pixel values to 0–1

Model Architecture

CNN Model

Input: 224×224×3
Layers: Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Softmax
Activation: ReLU (hidden), Softmax (output)
Optimizer: Adam
Loss: Categorical Crossentropy

MobileNet Model

Pretrained MobileNet (TensorFlow)
Fine-tuned last dense layer for 6 classes
Freezes base layers, trains top classification layer
Optimizer: Adam, Loss: Categorical Crossentropy

Training Details

Framework: TensorFlow 2.x
Data augmentation:
Random rotation, zoom, horizontal flip
Epochs: 20–30 (depending on dataset size)
Batch size: 32
Early stopping and model checkpoint used

Evaluation Metrics

Metric	CNN	MobileNet
Accuracy	0.92	0.96
Precision	0.91	0.95
Recall	    0.90	0.94
F1-score	0.90	0.94
Confusion matrices generated for both models
Grad-CAM used to interpret model attention on images

Usage – Streamlit GUI

Download models from kaggle
Use

import kagglehub

# Download latest version
path = kagglehub.model_download("sakshyamkarki/waste-image-classification/tensorFlow2/default")

print("Path to model files:", path)

Run FastAPI backend:
uvicorn app:app --reload --host 0.0.0.0 --port 8000

Run Streamlit frontend:

streamlit run streamlit_app.py

Upload an image, select CNN or MobileNet, click Predict

Output:

Predicted class
Confidence score
Grad-CAM heatmap

Backend API

Endpoint	Method	Description
/predict/cnn	POST	Predict with CNN, returns Grad-CAM image, predicted class in headers
/predict/mobilenet	POST	Predict with MobileNet, returns Grad-CAM image, predicted class in headers
/	GET	Root endpoint, welcome message

Grad-CAM Visualization

Highlights regions in the image contributing to the model’s decision
Useful for interpretability and understanding misclassifications

Project Deliverables

app.py – FastAPI backend
streamlit_app.py – Streamlit frontend
Trained models: cnn_waste_model.h5, mobilenet_waste_model.h5
Grad-CAM visualizations (generated dynamically)
Dataset description and links
README + evaluation metrics tables

Comments and Best Practices

Preprocessing ensures consistent input shape
Confidence score allows thresholding for uncertain predictions
Streamlit GUI provides intuitive user interface
Modular API endpoints allow easy extension (evaluation, batch processing)
Grad-CAM enhances explainability for model predictions

References

Kaggle Dataset: Waste Image Classification
TensorFlow Documentation: https://www.tensorflow.org/
Grad-CAM: Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization” (2017)