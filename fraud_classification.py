"""
Credit card fraud classification via Random Forest
Dataset: Credit Card Fraud Detection (Kaggle, ULB Machine Learning Group)
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)

RANDOM_STATE = 42

print("Loading creditcard.csv...")
df = pd.read_csv("creditcard.csv")
print(f"Rows: {len(df):,}, Features: {df.shape[1]}")
print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}% of transactions)")

X = df.drop(columns=["Class"])
y = df["Class"]

# Scale Time and Amount (V1-V28 are already PCA components, roughly standardised)
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {len(X_train):,} rows ({y_train.sum()} fraud)")
print(f"Test:  {len(X_test):,} rows ({y_test.sum()} fraud)")

print("\nTraining Random Forest (class_weight='balanced')...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Test set results ---")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4))
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision (AUPRC): {avg_precision:.4f}")
print("Confusion matrix:")
print(cm)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances.head(10))

# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Confusion matrix
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Legit", "Fraud"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Legit", "Fraud"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
axes[1].plot(rec, prec, label=f"AUPRC={avg_precision:.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[2].plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.3f}")
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[2].set_xlabel("False Positive Rate"); axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curve")
axes[2].legend()

plt.tight_layout()
plt.savefig("fraud_rf_results.png", dpi=150)
print("\nSaved plot: fraud_rf_results.png")

fig2, ax2 = plt.subplots(figsize=(7, 5))
importances.head(10).sort_values().plot(kind="barh", ax=ax2)
ax2.set_title("Top 10 feature importances (Random Forest)")
ax2.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("fraud_rf_feature_importance.png", dpi=150)
print("Saved plot: fraud_rf_feature_importance.png")

# --- Save results ---
results = {
    "dataset": "Credit Card Fraud Detection",
    "n_rows": int(len(df)),
    "n_fraud": int(df["Class"].sum()),
    "fraud_rate_pct": round(float(df["Class"].mean() * 100), 4),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "model": "RandomForestClassifier(n_estimators=200, class_weight='balanced')",
    "metrics": {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "average_precision_auprc": round(float(avg_precision), 4),
    },
    "confusion_matrix": cm.tolist(),
    "top_10_feature_importances": importances.head(10).round(4).to_dict(),
}
with open("fraud_rf_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results: fraud_rf_results.json")
