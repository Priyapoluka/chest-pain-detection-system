# 🫀 Real-Time Chest Pain & Fainting Detection System

![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-YOLOv8-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey)

An academic **machine learning** project that detects **chest pain and fainting episodes in real time** using a custom-trained YOLOv8 object detection model with a Flask web interface — enabling faster emergency response through visual AI monitoring.

> ⚠️ **Disclaimer:** This tool is intended for educational and portfolio purposes only and is not a substitute for professional medical advice or emergency services.

---

## 📌 Overview

This system continuously monitors video input to identify visual symptoms associated with chest pain and fainting — such as sudden collapse and body posture changes. It combines a fine-tuned YOLOv8 model with a health risk prediction system, accessible through a clean web interface.

---

## ✨ Features

- 🎯 Real-time fall & fainting detection from live video
- 📷 Image-based detection with visual bounding boxes
- 🎥 Video upload and real-time streaming detection
- 🧠 Health risk prediction (SpO2, Heart Rate, BP, Temperature, Resp Rate)
- 📊 Live health data charts
- 🔴 Risk level alerts — Low / Moderate / High
- 🧩 Custom attention mechanisms (CBAM) for improved accuracy
- ⚡ Lightweight inference using GhostConv modules

---

## 🧠 Model Architecture

Custom PyTorch modules integrated into YOLOv8:

- `Conv`, `Conv2`, `RepConv` — Standard & reparameterized convolutions
- `CBAM` — Convolutional Block Attention Module (Channel + Spatial Attention)
- `GhostConv` / `GhostBottleneck` — Lightweight feature generation
- `C2f`, `SPPF` — CSP Bottleneck & Spatial Pyramid Pooling
- `DFL` — Distribution Focal Loss head

---

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| Language | Python |
| Detection Model | YOLOv8 (Ultralytics) |
| ML Framework | PyTorch |
| Custom Modules | RepConv, CBAM, GhostConv |
| Backend | Flask |
| Frontend | HTML / CSS / JavaScript |
| Charts | Chart.js |
| Dataset | Roboflow (CC BY 4.0) |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Priyapoluka/chest-pain-detection-system.git
cd chest-pain-detection-system

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run the web application
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

---

## 📁 Project Structure

```
├── app.py               # Flask web application
├── train.py             # Model training script
├── conv_modules.py      # Custom PyTorch conv blocks
├── data.yaml            # Dataset configuration
├── requirements.txt     # Dependencies
├── static/              # Generated detection images
├── train/               # Training images
├── valid/               # Validation images
└── test/                # Test images
```

---

## 🖥️ Web Interface

The web app has two main sections:

**Health Monitoring Tab**
- Enter vitals manually or use Simulate
- Get instant risk prediction (Low / Moderate / High)
- View live line charts of health history

**Fall Detection Tab**
- Upload an image → get detection result with bounding boxes
- Upload a video → stream real-time fall detection

---

## 🛣️ Roadmap

- [x] Custom convolution modules (CBAM, RepConv, GhostConv)
- [x] YOLOv8 training pipeline
- [x] Flask web application
- [x] Health risk prediction system
- [x] Image & video detection
- [x] Real-time video streaming
- [ ] Improve detection accuracy
- [ ] Alert system (SMS/Email notifications)
- [ ] Deploy to cloud

---

## 📊 Dataset

This project uses a fall detection dataset from Roboflow:
- **License:** CC BY 4.0
- **Source:** [Roboflow Universe](https://universe.roboflow.com/jack603-naver-com/test-fy6a7/dataset/1)

---

## 👩‍💻 Author

**Priyapoluka** — B.Tech Graduate
📧 your-yagnapriya2409@gmail.com


---

## 📄 License

This project is for educational and portfolio purposes.
