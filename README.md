🐶 End-to-End Dog Vision Classifier (MobileNetV2 + TensorFlow)
This project is a complete end-to-end image classification pipeline that identifies dog breeds using deep learning and transfer learning with TensorFlow, TensorFlow Hub, and the MobileNetV2 feature extractor.

🚀 Project Overview
Goal: Classify images of dogs into one of 120 dog breeds using transfer learning.
Model: MobileNetV2 pre-trained on ImageNet (used as a frozen feature extractor).
Dataset: A subset of the Stanford Dogs Dataset.
Input size: 224x224 RGB images.
Evaluation: Accuracy on training, validation, and test sets.

📦 Features
Loads and preprocesses image data from directories
Converts image paths into tf.data.Dataset objects with batching, shuffling, and mapping
Uses tf.keras.Sequential and tf.keras.Model APIs
Adds custom output layers for classification
Trains and evaluates the model
Saves the trained model to Google Drive
Loads the saved model and makes predictions on new data
Visualizes predictions and prediction confidence

🛠️ Tools & Libraries
Python 3.x
TensorFlow 2.x
TensorFlow Hub
NumPy
Matplotlib
Google Colab (with GPU support)
Google Drive (for storage)

🧠 Key Concepts Covered
Image preprocessing and resizing with TensorFlow
Batching and performance optimization using tf.data
Transfer learning with frozen feature extractors
Visualizing prediction probabilities
Saving and loading models in .h5 or .keras formats

✅ Results
Achieved high accuracy using MobileNetV2 as the base model
Successfully classified dog breeds with good confidence
Created clean visualizations of predictions and confidence levels

📌 How to Use
Clone the repo or download the notebook.
Upload the notebook to Google Colab.
Connect your Google Drive.
Set up training, validation, and test image folders.
Run the cells step by step.
Train, evaluate, and save the model.

📚 Credits
Based on TensorFlow and TensorFlow Hub official best practices
Inspired by community tutorials and practical deep learning workflows
