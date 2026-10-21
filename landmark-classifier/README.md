# Landmark Classification & Tagging for Social Media

> Udacity Introduction to Deep Learning Nanodegree (Core #4) — Course Project

## Overview

Build a landmark classifier that automatically predicts the location of images based on depicted landmarks. This solves the problem of tagging photos that lack GPS metadata on photo-sharing platforms.

## Project Structure

```
├── src/
│   ├── data.py           # Data loading, transforms, visualization
│   ├── model.py          # Custom 5-layer CNN architecture
│   ├── optimization.py   # Loss function and optimizer
│   ├── train.py          # Training loop, validation, testing
│   ├── transfer.py       # ResNet50 transfer learning setup
│   ├── predictor.py      # TorchScript export and inference
│   └── helpers.py        # Utility functions
├── cnn_from_scratch.ipynb    # Part 1: Train CNN from scratch (≥50%)
├── transfer_learning.ipynb   # Part 2: Transfer learning with ResNet50 (≥60%)
├── app.ipynb                 # Part 3: Export and deploy model
├── requirements.txt
└── README.md
```

## Dataset

50 classes of world landmarks. Place the dataset in `landmark_images/` with `train/`, `valid/`, `test/` subdirectories.

## Setup

```bash
pip install -r requirements.txt
# Run notebooks in order: Part 1 → Part 2 → Part 3
```
