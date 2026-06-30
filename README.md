---
title: Smart Traffic Vision
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Smart Traffic Vision 🚦

A production-ready end-to-end Computer Vision system for traffic object detection, built with Streamlit and YOLOv8.

## Overview

This project detects objects in street images (e.g., cars, pedestrians, trucks, buses, motorcycles, bicycles, traffic lights, and stop signs). It acts as a portfolio-grade MLOps project that demonstrates model integration from Hugging Face Hub and a frontend UI.

## Installation & Running Locally

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the Frontend
streamlit run app.py
```
