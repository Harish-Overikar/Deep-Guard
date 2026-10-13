# 🛡️ Deep-Guard
### AI-Generated Face Detector

Deep-Guard is a deep learning application that classifies facial images as **Real** or **AI-Generated / Deepfake** using DenseNet121 transfer learning.

Built with TensorFlow, Keras, and Streamlit, it provides a simple web interface where users can upload a face image and view the model's prediction and confidence score.

> **Note:** Deep-Guard is designed specifically for facial images. It is not a general-purpose AI-generated image detector.

## ✨ Features

- **AI-generated face detection:** Classifies face images as Real or AI-Generated / Deepfake.
- **Deep learning:** Uses DenseNet121 with ImageNet-pretrained weights.
- **Transfer learning:** Adapts pretrained visual features to binary face classification.
- **Interactive web interface:** Upload an image and view the prediction using Streamlit.
- **Confidence score:** Displays the model's confidence in its predicted class.
- **Custom classification threshold:** Uses a 0.4 threshold for the Fake class.
- **Image preview:** Displays the uploaded image before analysis.

## 🧠 Model Architecture

Deep-Guard uses DenseNet121 as its feature extractor, followed by a custom binary classification head.

| Component | Details |
|---|---|
| Model | DenseNet121 |
| Pretrained weights | ImageNet |
| Deep learning framework | TensorFlow / Keras |
| Input dimensions | 128 × 128 pixels |
| Input channels | 3 (RGB) |
| Output layer | Sigmoid |
| Classification type | Binary classification |
| Decision threshold | 0.4 |
| Reported test accuracy | 90.92% |

### How it works

1. **Image upload:** The user uploads a facial image.
2. **Preprocessing:** The image is converted to RGB, resized to 128 × 128 pixels, and normalized.
3. **Feature extraction:** DenseNet121 extracts visual features from the image.
4. **Classification:** The model produces a score for the Fake class.
5. **Prediction:** The score is compared with the decision threshold to classify the image as Real or Fake.
6. **Result display:** The application displays the predicted class and its associated model confidence score.

The model uses the following class mapping:

- `0` — Real
- `1` — Fake

The current prediction threshold is `0.4`. A score above this threshold is classified as Fake; otherwise, it is classified as Real.

## 📊 Model Performance

The trained DenseNet121 model achieved a reported **90.92% test accuracy** during evaluation.

This result reflects performance on the project's evaluated test dataset. It does not guarantee the same accuracy on unseen images, different datasets, or newer AI face-generation methods.

## 🛠️ Tech Stack

- **Python** — Core programming language
- **TensorFlow** — Deep learning framework
- **Keras** — Model development and inference
- **DenseNet121** — Pretrained convolutional neural network
- **NumPy** — Numerical operations and image arrays
- **Pillow (PIL)** — Image loading and preprocessing
- **Streamlit** — Interactive web application

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Harish-Overikar/Deep-Guard.git
cd Deep-Guard
```

### 2. Create a virtual environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```


### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will launch the application in your default browser.

## 📁 Project Structure

```text
Deep-Guard/
│
├── app.py
├── predict.py
├── requirements.txt
├── README.md
├── deepfake_model.keras
├── Testing_Images/
└── DeepFake_Image_Detector.ipynb
```

- `app.py` — Streamlit user interface
- `predict.py` — Model loading, image preprocessing, and prediction logic
- `requirements.txt` — Python dependencies
- `deepfake_model.keras` — Trained Keras model
- `Testing_Images/` — Optional sample images for local testing
- `DeepFake_Image_Detector.ipynb` — Notebook containing model experimentation

The testing images and notebook are optional repository contents.


## 🎯 Project Objective

Deep-Guard explores how deep learning and transfer learning can be applied to the detection of AI-generated facial images.

The project combines a pretrained convolutional neural network with a lightweight web interface to make model predictions accessible to users.

## 👨‍💻 Disclaimer

Deep-Guard is an educational and experimental project. Its predictions may be incorrect and should not be used as the sole basis for consequential decisions about a person's identity, authenticity, or trustworthiness.

---

**Deep-Guard** · AI-Generated Face Detector  
*Exploring facial image authenticity through deep learning.*
