# What Drives California Housing Prices?

## Motivation

This project analyzes the California housing dataset using the CRISP-DM process for the Udacity Data Scientist Blog Post project. The goal is to answer practical business questions about housing value drivers and communicate the results for non-technical stakeholders.

## Files

- `california_housing_blog_analysis.ipynb`: Executable Jupyter notebook containing data gathering, assessment, cleaning, analysis, modeling, and visualizations.
- `data/housing.csv`: California housing data downloaded from the public Hands-On ML repository.
- `images/`: Visualizations used in the notebook and blog post.
- `blog_post.md`: Stakeholder-facing article.
- `analysis_summary.json`: Reproducible summary metrics from the analysis.

## Libraries

The project uses Python 3, pandas, NumPy, scikit-learn, matplotlib, and seaborn.

## Questions Answered

1. How complete is the dataset, and how should missing bedroom values be handled?
2. How does median income relate to median house value?
3. How much does ocean proximity matter?
4. Can a transparent baseline model provide useful price predictions?

## Summary of Results

- The dataset has 20,640 rows and 207 missing `total_bedrooms` values.
- Median home value rises strongly across income quartiles.
- Ocean and bay-adjacent locations have higher median values than inland locations.
- The baseline linear model achieved RMSE $72,188, MAE $51,845, and R² 0.613 on a holdout set.

## Acknowledgements

The dataset is the California housing dataset hosted in the Hands-On Machine Learning public repository.
