# Image Classification with Pre-trained ResNet-50
[![Deep Learning](https://img.shields.io/badge/Domain-Computer%20Vision-blue)](https://en.wikipedia.org/wiki/Computer_vision)
[![Model](https://img.shields.io/badge/Architecture-ResNet--50-red)](https://arxiv.org/abs/1512.03385)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)

## 🏗️ Project Overview
This repository focuses on **Transfer Learning**, one of the most practical and powerful techniques in modern Deep Learning. Instead of training a model from scratch, this project utilizes **ResNet-50**—a 50-layer deep residual network—pre-trained on the massive **ImageNet** dataset. This project demonstrates how to ingest, preprocess, and classify images using this sophisticated architecture to achieve professional-grade accuracy with minimal computational overhead.

---

## 🛠️ Key Technical Implementations

### 1. Residual Network Architecture
* **Identity Mapping:** Utilizing the "Shortcut Connections" inherent in ResNet that allow gradients to flow through very deep networks without vanishing.
* **Feature Extraction:** Leveraging the 25.6 million parameters of ResNet-50 that have already learned to identify complex shapes, textures, and object parts.

### 2. High-Fidelity Preprocessing
* **Input Standardization:** Transforming raw images into the $224 \times 224$ format required by the model.
* **Preprocessing Pipeline:** Utilizing `resnet50.preprocess_input` to ensure the input data distribution matches the original training distribution (Zero-centering and RGB-to-BGR conversion).

### 3. Inference & Decoding
* **Top-K Prediction:** Implementing logic to extract the most likely class labels from the softmax output.
* **Result Interpretation:** Decoding numerical indices into descriptive labels (e.g., "Golden Retriever", "Sports Car") using the ImageNet label map.

---

## 💻 Tech Stack
* **Language:** Python 3.9+
* **Deep Learning:** TensorFlow / Keras
* **Image Processing:** PIL (Pillow), NumPy
* **Environment:** Jupyter Notebook

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/transfer-learning-resnet50-predictions.git](https://github.com/your-username/transfer-learning-resnet50-predictions.git)

Install Dependencies:

Bash
pip install tensorflow numpy pillow matplotlib


3. **Run the Notebook:**
   Open `using pretrained models.ipynb` and upload your own images to see the ResNet-50 predictions in action.
   
