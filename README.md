Project Title

Smart Waste Image Classifier – CNN & MobileNet with Streamlit GUI

Objective

Build an image classification system that classifies waste images into the following categories:

Cardboard
Glass
Metal
Paper
Plastic
Trash

The system uses CNN and MobileNet models and provides confidence scores.
A Streamlit GUI allows easy image uploads and predictions.

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
Epochs: 10
Batch size: 32

Evaluation Metrics

Metric	    CNN	    MobileNet
Accuracy	0.92	0.96
Precision	0.91	0.95
Recall	    0.90	0.94
F1-score	0.90	0.94
Confusion matrices generated for both models

Usage – Streamlit GUI

Download models from kaggle

import kagglehub

# Download latest version
path = kagglehub.model_download("sakshyamkarki/waste-image-classification/tensorFlow2/default")

print("Path to model files:", path)

Run Streamlit frontend:

streamlit run waste_classifier.py

Upload an image, select CNN or MobileNet, click Predict

Output:

Predicted class
Confidence score

Project Deliverables

waste_classifier.py – Streamlit
Trained models: cnn_waste_model.h5, mobilenet_waste_model.h5
Dataset description and links
README + evaluation metrics tables

References

Kaggle Dataset: Waste Image Classification
TensorFlow Documentation: https://www.tensorflow.org/