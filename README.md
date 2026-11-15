# 🎙️ Vocal Emotion Analytics Engine (Speech Emotion Recognition SaaS)

A production-grade, enterprise-ready Speech Emotion Recognition (SER) web application built with **Streamlit** and a **TensorFlow CNN-LSTM** architecture. It performs real-time audio feature extraction and provides executive telemetry analytics dashboards.

---

## 🚀 Key Features
* **Real-Time Audio Capture:** Direct browser microphone recording with live helper captions.
* **Deep Learning Inference:** TensorFlow/Keras CNN-LSTM model trained to classify complex human emotions from vocal patterns.
* **Acoustic Feature Engineering:** Extracts 40-dimensional Mel-Frequency Cepstral Coefficients (MFCCs) using Librosa.
* **Telemetry & Analytics Dashboards:** Visualizes confidence scores, probability distributions, and session history metrics.
* **Session Audit & Export:** Built-in session log with a strict entry limit and CSV/JSON data export capabilities.
* **Secure Access:** Integrated authentication gate for enterprise environments.

---

## 🛠️ Project Structure
```text
vocal-emotion-analytics-engine/
│
├── dataset/raw/          # Raw audio datasets and processing scripts
├── models/               # Trained TensorFlow/Keras model binaries & weights
├── outputs/              # Exported audit logs and telemetry data
├── samples/              # Sample audio clips for testing inference
├── src/                  # Core Streamlit application and utility modules
└── requirements.txt      # Project Python dependencies
