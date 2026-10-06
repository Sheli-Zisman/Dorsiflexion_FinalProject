---
title: Dorsiflexion Measurement
emoji: 🦶
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Dorsiflexion Measurement

An AI-powered web application for estimating ankle dorsiflexion range of motion from lateral leg images using computer vision and deep learning.

## Project Team

Developed as a Software Engineering final project by:

- Ron Kroitoro
- Sheli Zisman

The project was developed in collaboration with medical professionals and focuses on providing an automated alternative for ankle dorsiflexion measurement.

## Project Overview

The system receives two lateral images of the same leg: one in a resting position and one in maximum dorsiflexion.

A customized DeepLabCut model automatically detects anatomical landmarks in each image. The detected keypoints are then used to calculate the ankle angle in both positions and determine the final dorsiflexion range of motion.

## Technologies

- Python
- DeepLabCut
- ResNet-50
- OpenCV
- Flask
- REST API
- JavaScript
- Docker

## Model Performance

- Test mAP: 94.48%
- Test mAR: 96.76%
- Test RMSE: 8.05 px

## Architecture

The application consists of a web-based frontend, a Python/Flask backend, a DeepLabCut-based computer vision pipeline, REST API communication, and Docker-based deployment.
