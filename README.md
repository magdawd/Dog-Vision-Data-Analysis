# Dog-Vision-Data-Analysis

This project builds a convolutional neural network to classify images of dog breeds using transfer learning with MobileNetV2. It demonstrates loading data, preprocessing, training, and evaluating a deep learning model using TensorFlow and Keras.

## Project Overview
- Uses MobileNetV2 as a base model for transfer learning
- Implements custom classification layers on top of the pre-trained model
- Trains on labeled dog image data organized by breed
- Visualizes performance with accuracy/loss plots and confusion matrix

## Model Architecture
Base Model: MobileNetV2 (pre-trained on ImageNet, used without top layer)
Dense layer(s) with ReLU
Output layer with Softmax activation

## Training and Evaluation
- The model is trained using categorical cross-entropy loss and Adam optimizer
- Accuracy and loss are tracked over epochs

Final performance is evaluated using:
- Accuracy score
- Sample prediction visualization

## Requirements
Python 3.x
TensorFlow
NumPy
Matplotlib
Scikit-learn
Pandas

This code was provided as part of a course "Complete A.I. & Machine Learning, Data Science Bootcamp" by Andrei Neagoie and Daniel Bourke. 
Working through it helped me understand key concepts and techniques in machine learning.
