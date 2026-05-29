# 🫀 Real-Time Chest Pain & Fainting Detection System

![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-YOLOv8-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)

An academic **machine learning** project that detects **chest pain and fainting episodes in real time** using a custom-trained YOLOv8 object detection model — enabling faster emergency response through visual AI monitoring.

> ⚠️ **Disclaimer:** This tool is intended for academic purposes only and is not a substitute for professional medical advice or emergency services.

---

## 📌 Overview

This system continuously monitors video input to identify visual symptoms associated with chest pain and fainting — such as sudden collapse, body posture changes, and distress indicators. It uses a fine-tuned YOLOv8 model enhanced with custom attention-based convolutional modules for improved detection accuracy.

---

## 🧠 Model Architecture

Custom PyTorch modules integrated into YOLOv8:

- `Conv`, `Conv2`, `RepConv` — Standard & reparameterized convolutions
- `CBAM` — Convolutional Block Attention Module (Channel + Spatial Attention)
- `GhostConv` / `GhostBottleneck` — Lightweight feature generation
- `C2f`, `C3`, `SPPF` — CSP Bottleneck & Spatial Pyramid Pooling
- `DFL` — Distribution Focal Loss head

---

## ✨ Features

- 🎯 Real-time chest pain & fainting detection from live video
- 🧩 Custom attention mechanisms (CBAM) for improved accuracy
- ⚡ Lightweight inference using GhostConv modules
- 🖥️ User interface prototype
- 🗄️ SQL-based data logging

---

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| Language | Python |
| Detection Model | YOLOv8 (Ultralytics) |
| ML Framework | PyTorch |
| Custom Modules | RepConv, CBAM, GhostConv |
| Frontend | HTML / CSS |
| Database | SQL |

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

# Run real-time detection
python app.py
```

---

## 📁 Project Structure

```
├── data.yaml          # Dataset configuration
├── train.py           # Model training script
├── conv_modules.py    # Custom PyTorch conv blocks
├── requirements.txt   # Dependencies
├── static/            # CSS and assets
├── templates/         # HTML files
└── app.py             # Main application (real-time detection)
```

---

## 🛣️ Roadmap

- [x] Custom convolution modules (CBAM, RepConv, GhostConv)
- [x] YOLOv8 training pipeline
- [ ] Improve detection accuracy
- [ ] Real-time video stream integration
- [ ] Alert system for detected events
- [ ] Deploy as web application

---

## 👩‍💻 Author

**Priyapoluka** — Academic Project

---

## 📄 License

This project is for academic use only.
