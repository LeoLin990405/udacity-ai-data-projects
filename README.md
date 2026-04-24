# Text Translation & Sentiment Analysis using Transformers

Udacity **Deep Learning Nanodegree (nd101)** — Course 4: RNNs and Transformers, final project.

## Overview

Analyze sentiment of movie reviews in three languages (English, French, Spanish):

1. **Translation** — fr→en and es→en via HuggingFace `Helsinki-NLP/opus-mt-*` MarianMT models
2. **Merge** — combine the three CSVs into one dataframe
3. **Sentiment** — binary classification via DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)
4. **Export** — `result/reviews_with_sentiment.csv`

## Results

30 reviews (10 per language) → **15 Positive / 15 Negative**.

## Stack

- Python 3.14 / PyTorch 2.10 / HuggingFace transformers 5.3
- Runs on CPU or MPS (Apple Silicon); no GPU required

## Run

```bash
pip install transformers sentencepiece sacremoses pandas torch
python run_project.py   # inference-only, ~2-3 min
```

## Files

- `project.ipynb` / `project.html` — executed submission notebook
- `run_project.py` — standalone training/inference script
- `data/movie_reviews_{eng,fr,sp}.csv` — source reviews
- `result/reviews_with_sentiment.csv` — final output

## Context

Part of Leo's [Udacity Masters in AI](https://github.com/LeoLin990405?tab=repositories) journey. See also:
- [landmark-classifier](https://github.com/LeoLin990405/landmark-classifier) — Course 3 CNN project
- [face-generation](https://github.com/LeoLin990405/face-generation) — Course 5 DCGAN project
