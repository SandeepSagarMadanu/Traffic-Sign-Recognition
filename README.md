# Traffic Sign Image Classification using CNNs

## 📌 Project Overview

This project focuses on developing and evaluating a Convolutional Neural Network (CNN) model to classify traffic sign images with high accuracy. The goal is to assist in the development of autonomous driving systems and smart traffic management by enabling real-time and accurate identification of road signs.

This repository contains:
- A detailed report outlining the methodology and findings.
- Jupyter notebooks for training and evaluating CNN models.
- Performance evaluation metrics and visualizations.

---

## 🧠 Problem Statement

With the increasing development of intelligent transportation systems and autonomous vehicles, accurately recognizing traffic signs is essential. Manual detection is inefficient and error-prone; hence, machine learning, particularly deep learning, offers a powerful solution.

---

## 📁 Dataset

The dataset used for this project consists of color images of 43 different traffic signs, organized into corresponding class folders. The dataset contains:

- 43 classes of traffic signs
- 39,209 training images
- 12,630 testing images

Each image is a 32x32 pixel RGB image. The dataset is publicly available and often used for benchmarking traffic sign classification tasks.

---

## 🧰 Tools and Technologies

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Jupyter Notebook

---

## 🏗️ Model Architecture

The Convolutional Neural Network (CNN) includes:
- Convolutional layers for feature extraction
- ReLU activation functions
- MaxPooling for downsampling
- Dropout layers to prevent overfitting
- Fully connected dense layers for classification
- Softmax output layer for multi-class classification

Key configurations:
- Input shape: (32, 32, 3)
- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

---

## 🚀 Training Process

- Images were normalized and one-hot encoded.
- Data augmentation was applied to improve generalization.
- The model was trained over several epochs using the training set and validated on a separate validation set.

---

## 📊 Evaluation and Results

- Achieved **94-97% accuracy** on the test set.
- Plotted training vs. validation loss and accuracy curves.
- Confusion matrix and classification report were generated to evaluate per-class performance.

Visualizations include:
- Loss and accuracy plots
- Confusion matrix heatmap
- Sample predictions

---

## 🛠️ Requirements

Install dependencies using pip:

pip install tensorflow numpy matplotlib seaborn pandas scikit-learn

## Future Work
- Integrate model into a real-time traffic sign detection system.

- Improve generalization to work with varied lighting and occlusion conditions.

- Deploy as a web or mobile application.

## 🙏 Acknowledgments
- Dataset: German Traffic Sign Recognition Benchmark (GTSRB)

- TensorFlow and Keras documentation

- Open-source community contributions
