# Image Classification Using Pre-trained ResNet-50

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Deep Learning](https://img.shields.io/badge/Domain-Computer%20Vision-purple)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Keras](https://img.shields.io/badge/Library-Keras-red)
![Model](https://img.shields.io/badge/Model-ResNet--50-green)
![Status](https://img.shields.io/badge/Status-Learning%20Project-brightgreen)

A practical Computer Vision project demonstrating **image classification using a pre-trained ResNet-50 model** with TensorFlow and Keras. It explores transfer learning, image preprocessing, ImageNet class prediction, and how pre-trained deep learning models can be used to recognize objects without training a neural network from scratch.

---

## 📌 Project Overview

Training a deep learning model from scratch requires a large dataset, computational resources, and considerable training time.

Transfer Learning provides an alternative by allowing us to reuse knowledge learned by a model trained on a large dataset.

In this project, I explore **ResNet-50**, a deep Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset.

The implementation demonstrates how to:

- Load a pre-trained ResNet-50 model using Keras.
- Prepare input images for model inference.
- Apply the required ResNet-50 preprocessing.
- Generate predictions for input images.
- Decode prediction scores into meaningful ImageNet class labels.
- Explore Top-K classification results.

The project focuses on understanding and using an existing pre-trained model rather than training a new model from scratch.

## 🎯 Project Objectives

- Understand the fundamentals of Transfer Learning.
- Explore the ResNet-50 architecture.
- Learn how pre-trained ImageNet models work.
- Understand image resizing and preprocessing.
- Perform image classification using a pre-trained CNN.
- Interpret model predictions and confidence scores.
- Gain practical experience with TensorFlow and Keras.

## 🧠 Understanding ResNet-50

**ResNet-50 (Residual Network with 50 layers)** is a deep CNN architecture introduced to make training deeper neural networks more effective.

One of its important features is the use of **residual connections**, also known as shortcut connections.

These connections allow information and gradients to pass through the network more directly, helping address optimization difficulties in very deep networks.

### Why use a pre-trained ResNet-50?

The model has already learned useful visual features from the ImageNet dataset, including:

- Edges and textures
- Shapes and patterns
- Object parts
- Higher-level visual representations

Instead of learning these features from the beginning, we can reuse the pre-trained model to classify new images.

## 🏗️ Project Workflow

The image classification workflow follows these steps:

```text
       Input Image
            |
            v
      Load Image
            |
            v
    Resize to 224 × 224
            |
            v
   ResNet-50 Preprocessing
            |
            v
   Pre-trained ResNet-50
            |
            v
     Prediction Scores
            |
            v
       Top-K Results
            |
            v
    ImageNet Class Labels
```

## 🛠️ Key Technical Implementations

### 1. Loading the Pre-trained Model

The project uses the ResNet-50 architecture available through TensorFlow/Keras.

The model comes with weights pre-trained on ImageNet, allowing it to recognize 1,000 ImageNet classes.

```python
from tensorflow.keras.applications import ResNet50

model = ResNet50(weights="imagenet")
```

This loads the model with its pre-trained ImageNet weights and classification head.

### 2. Image Preprocessing

Deep learning models expect input images in a particular format.

For ResNet-50, the image is prepared using the expected input dimensions and preprocessing function.

The standard input dimensions are:

```text
224 × 224 × 3
```

The preprocessing function is:

```python
from tensorflow.keras.applications.resnet50 import preprocess_input

image = preprocess_input(image)
```

The ResNet-50 preprocessing function adjusts the input values to match the format expected by the model, including channel conversion and mean-centering.

**Important:** Correct preprocessing is essential for obtaining meaningful predictions from a pre-trained model.

### 3. Image Classification

After preprocessing, the image is passed through the model to generate classification scores.

```python
predictions = model.predict(image)
```

The output contains scores corresponding to the ImageNet classes.

### 4. Decoding Predictions

The numerical prediction scores can be converted into readable class labels using Keras' ImageNet decoding utility.

```python
from tensorflow.keras.applications.resnet50 import decode_predictions

results = decode_predictions(predictions, top=5)
```

This returns the top five predicted ImageNet classes along with their associated scores.

For example, an image of a cat may produce predictions related to different cat breeds.

The actual predictions depend on the input image.

## 📊 What This Project Demonstrates

The notebook provides a practical introduction to:

| Concept | Description |
|---|---|
| Transfer Learning | Reusing knowledge from a pre-trained model |
| ResNet-50 | Deep residual CNN architecture |
| ImageNet | Large-scale image classification dataset |
| Image Preprocessing | Preparing images for model inference |
| Feature Extraction | Using learned visual representations |
| Image Classification | Predicting image categories |
| Top-K Prediction | Displaying the most likely classes |

## 💻 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep Learning framework |
| Keras | Pre-trained model implementation |
| NumPy | Numerical operations |
| Pillow (PIL) | Image loading and processing |
| Matplotlib | Image visualization |
| Jupyter Notebook | Interactive development environment |

## 📁 Repository Structure

```text
using-pretrained-models/
│
├── using-pretrained-models.ipynb
│   └── ResNet-50 implementation and image classification
│
├── cat pic.jpg
│   └── Sample image for classification
│
└── README.md
    └── Project documentation
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mani787060/using-pretrained-models.git

cd using-pretrained-models
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install tensorflow numpy pillow matplotlib jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

`using-pretrained-models.ipynb`

Run the notebook cells sequentially to explore the ResNet-50 implementation.

### 5. Try Your Own Images

You can experiment with other images by updating the image path in the notebook.

For meaningful predictions, use images containing objects that belong to the ImageNet class categories.

## ⚙️ Important Notes

- The notebook uses pre-trained ImageNet weights.
- The first execution may download model weights.
- Internet access may be required for the initial download.
- ResNet-50 inference can run on a CPU, although it may be slower than GPU inference.
- The model predicts among its ImageNet classes; it is not automatically trained to recognize arbitrary custom categories.
- This implementation demonstrates inference using a pre-trained model. It does not establish that custom fine-tuning or additional model training has been performed.

## 🔮 Future Improvements

Some possible extensions include:

- Compare ResNet-50 with VGG16, MobileNetV2, and EfficientNet.
- Display prediction confidence scores in a visual interface.
- Build a Streamlit-based image classification application.
- Experiment with feature extraction for custom datasets.
- Fine-tune the model on a custom image classification dataset.
- Evaluate inference time and model size.
- Explore data augmentation and model evaluation techniques.

## 🎓 Key Learning Outcomes

Through this project, I explored:

- Transfer Learning in Deep Learning.
- ResNet architecture and residual connections.
- Loading pre-trained models with Keras.
- Image preprocessing for CNN architectures.
- ImageNet-based image classification.
- Top-K predictions and label decoding.
- Practical applications of pre-trained Computer Vision models.

## 👨‍💻 Author

**Mani Ratan**

- GitHub: [@mani787060](https://github.com/mani787060)
- LinkedIn: [Mani Ratan](https://www.linkedin.com/in/mani-ratan-159963289/)
- Kaggle: [@mani787060](https://www.kaggle.com/mani787060)

This repository is part of my continuous learning journey in Deep Learning, Computer Vision, and Artificial Intelligence.

⭐ If you find this project useful, consider giving the repository a star!
