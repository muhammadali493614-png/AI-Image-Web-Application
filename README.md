# AI Image Web Application — PPE Detection

AI-powered web application for PPE detection using deep learning, YOLOv8, computer vision, and Flask.

## Overview

AI Image Web Application is a web-based computer vision project focused on Personal Protective Equipment (PPE) detection.

The application provides a web interface for processing images and camera input and detecting PPE-related objects using an AI-based object detection model.

## Key Features

- PPE detection using YOLOv8
- Image-based object detection
- Webcam-based detection
- Real-time detection support
- Flask web application
- Computer vision processing
- Detection result generation
- Database integration
- User authentication and security features
- Detection reporting functionality

## Technologies Used

- Python
- Flask
- YOLOv8
- Ultralytics
- OpenCV
- Computer Vision
- Deep Learning
- HTML
- CSS
- JavaScript
- SQLite

## PPE Detection

The application is designed to detect PPE-related objects using computer vision and deep learning.

The detection classes depend on the trained PPE detection model used by the application.

## Project Structure

```text
AI-Image-Web-Application/
│
├── database/
│   ├── __init__.py
│   ├── db_helper.py
│   └── schema.sql
│
├── models/
│   ├── face_labels.json
│   ├── face_recognizer.yml
│   ├── predict.py
│   └── train.py
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   ├── index.html
│   └── reset_password.html
│
├── utils/
│   ├── email_notifier.py
│   ├── face_recognition_helper.py
│   ├── model_validation.py
│   ├── report_generator.py
│   └── security.py
│
├── app.py
├── config.py
├── requirements.txt
├── SECURITY_SETUP.md
├── webcam_test.py
└── README.md