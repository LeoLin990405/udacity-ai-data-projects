# Recommendations with IBM

This repository contains a completed recommendation system project for IBM Watson Studio community articles.

## What Is Included

- `Recommendations_with_IBM.ipynb`: Completed notebook with EDA, rank-based recommendations, user-user collaborative filtering, content-based recommendations, and SVD article similarity.
- `data/user-item-interactions.csv`: User-article interaction data.
- `data/articles_community.csv`: Article content metadata.
- `project_tests.py`, `top_5.p`, `top_10.p`, `top_20.p`: Provided project tests and expected top article fixtures.
- `svd_accuracy.png`: SVD reconstruction accuracy plot used to discuss latent feature selection.
- `metrics_summary.json`: Summary values and sample recommendation outputs.

## Methods

The notebook implements four recommendation approaches:

1. Rank-based recommendations for new users.
2. User-user collaborative filtering with similarity and popularity tie-breakers.
3. Content-based TF-IDF and KMeans article similarity.
4. Matrix factorization with SVD and cosine similarity in latent article space.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Recommendations_with_IBM.ipynb
```
