# Data Science Pipeline Project

This repository contains a completed StyleSense recommendation prediction pipeline for the Udacity Data Scientist Nanodegree.

## Project Summary

The project predicts the `Recommended IND` label from customer clothing reviews. The model uses a single scikit-learn pipeline that handles:

- Numerical features: `Age` and `Positive Feedback Count`.
- Categorical features: `Clothing ID`, `Division Name`, `Department Name`, and `Class Name`.
- Text features: `Title` and `Review Text`, using normalized tokenization, light stemming, TF-IDF, and custom text statistics.

## Files

- `Data_Science_Pipeline.ipynb`: The completed notebook with data loading, preprocessing, model pipeline creation, grid search tuning, and final test-set evaluation.
- `data/reviews.csv`: The project dataset copied from the starter repository.
- `metrics_summary.json`: Final model metrics and selected hyperparameters from the executed training run.
- `starter/`: Original starter materials retained for reference.
- `requirements.txt`: Python package dependencies.

## Final Metrics

- Accuracy: `0.8897`
- Precision: `0.9680`
- Recall: `0.8944`
- Best CV F1: `0.9264`

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Data_Science_Pipeline.ipynb
```

The notebook keeps preprocessing inside the scikit-learn pipeline so inference uses the same transformations as training.
