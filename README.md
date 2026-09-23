# Hand Gesture Recognition - Transformer Encoder

A real-time hand gesture recognition system built with **MediaPipe Keypoint Extraction** and a **Transformer Encoder model**, using **TorchScript** for inference and **SocketIO** for real-time streaming.

---

## 🚀 Features
- Real-time hand keypoint extraction
- Transformer Encoder–based classifier
- TorchScript model export
- SocketIO backend for low-latency streaming predictions
- Handles dynamic sequences of hand gestures

---

## 🧠 Model Architecture
- **Input:** 21×3 hand keypoints per hand  
- **Backbone:** Transformer Encoder (multi-head attention)  
- **Classifier:** Fully connected layers  
- **Output:** Gesture class probabilities

---

## ⚙️ Installation
```bash
pip install -r requirements.txt
```
## Running Inference
```bash
python backend/main.py







