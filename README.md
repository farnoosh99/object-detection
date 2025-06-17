# YOLOv8 Object Detection App

A powerful and easy-to-use Python application for object detection using the YOLOv8 deep learning model. This tool allows users to detect objects in **images**, **video files**, and **live webcam** streams, with options to customize detection settings.

## 🚀 Key Features

- ✅ Choose YOLOv8 model size: `n`, `m`, or `l`
- ✅ Adjustable **Confidence** and **IoU** thresholds
- ✅ Supports object detection on:
  - 🖼️ Images (with automatic output saving)
  - 🎬 Video files (with real-time overlay and export)
  - 📷 Live webcam streams (with recording option)
- ✅ Intuitive GUI powered by EasyGUI
- ✅ Lightweight, no complex setup required

## 📦 Requirements

- Python 3.10+
- `ultralytics`
- `opencv-python`
- `torch`, `torchvision`, `torchaudio`
- `easygui`

Install all dependencies using:

```bash
pip install ultralytics opencv-python torch torchvision torchaudio easygui
