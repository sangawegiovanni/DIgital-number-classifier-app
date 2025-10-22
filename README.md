# 🧠 Digital Number Classification App

A lightweight **FastAPI-based** web service that classifies handwritten digits (0–9) from uploaded images using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. This project demonstrates the practical use of **deep learning for computer vision**, integrating a trained model with an efficient **FastAPI backend** for real-time predictions.

---

## 🚀 Features

* 🖼️ Accepts `.png` or `.jpg` image uploads of handwritten digits.
* ⚡ Fast and asynchronous **FastAPI** backend for quick responses.
* 🧩 CNN model trained on **MNIST** for accurate digit recognition.
* 🔍 Returns predicted digit in JSON format.
* 🧠 Easily extendable for deployment or frontend integration (e.g., React, Streamlit, or mobile).

---

## 🧰 Tech Stack

| Component         | Technology                         |
| ----------------- | ---------------------------------- |
| **Backend**       | FastAPI                            |
| **Deep Learning** | PyTorch                            |
| **Dataset**       | MNIST                              |
| **Model Type**    | Convolutional Neural Network (CNN) |
| **Environment**   | Python 3.10+                       |

---

## 🧪 Model Overview

The CNN model was trained using the MNIST dataset (60,000 training and 10,000 testing images).
It consists of:

* Two convolutional layers for feature extraction
* ReLU activations for non-linearity
* Max pooling for dimensionality reduction
* Fully connected layers for classification

The model achieves **~99% accuracy** on the MNIST test set.

---

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/digital-number-classifier.git
   cd digital-number-classifier
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI app:**

   ```bash
   uvicorn main:app --reload
   ```

5. **Access the API:**
   Open your browser or API client and go to:

   ```
   http://127.0.0.1:8000/docs
   ```

   Here you can upload an image and view the predicted digit.

---

## 🧾 API Endpoints

| Method | Endpoint   | Description                                           |
| ------ | ---------- | ----------------------------------------------------- |
| `POST` | `/predict` | Upload an image (PNG/JPG) and get the predicted digit |
| `GET`  | `/`        | Welcome route (optional health check)                 |

**Example request (via cURL):**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-F "file=@sample_digit.png"
```

**Example response:**

```json
{
  "predicted_digit": 7
}
```

---

## 📁 Project Structure

```
digital-number-classifier/
│
├── model/
│   └── cnn_mnist.pth          # Trained model weights
│
├── main.py                    # FastAPI entry point
├── model_cnn.py               # CNN model architecture
├── predict.py                 # Image preprocessing and prediction logic
├── requirements.txt
└── README.md
```

---

## 🧩 Future Improvements

* Add frontend interface (React/Streamlit) for interactive predictions.
* Deploy on cloud (Render, Hugging Face Spaces, or AWS).
* Extend model to recognize handwritten letters (A–Z).

---

## 👨‍💻 Author

**Giovanni Sangawe**
Undergraduate Electronics Engineering Student, University of Dar es Salaam

> “Intelligence, Amplified.”

🔗 https://www.linkedin.com/in /giovannisangawe| [GitHub](#) | giovannisangawe@gmail.com(#)

---
