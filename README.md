# Case Studies in Data Science — Individual Task 1, Part 1.3: Data Analysis

Two independent machine learning pipelines, applied to the two datasets identified in Part 1.2,
for the RMIT Case Studies in Data Science assignment (Senior Consultant, Data Engineering role at Practiv).

## Datasets (not included — download separately)

1. **Online Retail II** (UCI Machine Learning Repository)
   https://archive.ics.uci.edu/dataset/502/online+retail+ii
   Download `online_retail_II.xlsx` and place it in this folder.

2. **Credit Card Fraud Detection** (Kaggle, ULB Machine Learning Group)
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   Download `creditcard.csv` and place it in this folder.

Both are excluded from version control (see `.gitignore`) due to file size.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

- `retail_segmentation.py` — cleans the Online Retail II transaction log, engineers RFM
  (Recency, Frequency, Monetary) features per customer, selects the number of clusters via
  silhouette score, and fits a KMeans model to segment customers.
  Outputs: `retail_kmeans_results.png`, `retail_kmeans_results.json`, `retail_rfm_with_clusters.csv`

- `fraud_classification.py` — trains a Random Forest classifier (`class_weight="balanced"`) on
  the Credit Card Fraud Detection dataset to flag fraudulent transactions, evaluated with metrics
  suited to extreme class imbalance (precision, recall, F1, ROC-AUC, AUPRC).
  Outputs: `fraud_rf_results.png`, `fraud_rf_feature_importance.png`, `fraud_rf_results.json`

Run either script directly once the corresponding data file is in place:

```bash
python retail_segmentation.py
python fraud_classification.py
```

## Results (already generated, included in this repo)

| Model | Metric | Value |
|---|---|---|
| KMeans (retail) | Silhouette score (k=2) | 0.42 |
| Random Forest (fraud) | Precision | 0.972 |
| Random Forest (fraud) | Recall | 0.709 |
| Random Forest (fraud) | F1 | 0.820 |
| Random Forest (fraud) | ROC-AUC | 0.933 |
| Random Forest (fraud) | Average Precision (AUPRC) | 0.812 |

See the JSON result files for full details, and the assignment report (Part 1.3) for interpretation.

## AI use disclosure

Code for both pipelines was drafted with AI assistance (Claude) under RMIT's Condition 3 Bounded
Process AI policy (code assistance for ML experiments is an approved use). Both scripts were run
to produce the real results above; the interpretation and analysis in the assignment report were
written independently. See the Condition 3 Bounded Process AI Declaration submitted with the
assignment for full details.
