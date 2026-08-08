"""
Retail customer segmentation via KMeans clustering (RFM features)
Dataset: Online Retail II (UCI Machine Learning Repository)
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

RANDOM_STATE = 42

print("Loading Online Retail II (both sheets)...")
sheets = pd.read_excel("online_retail_II.xlsx", sheet_name=["Year 2009-2010", "Year 2010-2011"])
df = pd.concat(sheets.values(), ignore_index=True)
print(f"Raw rows: {len(df):,}")

# --- Cleaning ---
df = df.dropna(subset=["Customer ID"])
df = df[~df["Invoice"].astype(str).str.startswith("C")]  # remove cancellations
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
df["TotalPrice"] = df["Quantity"] * df["Price"]
print(f"Rows after cleaning: {len(df):,}")
print(f"Unique customers: {df['Customer ID'].nunique():,}")

# --- RFM feature engineering ---
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalPrice", "sum"),
).reset_index()

print("\nRFM summary statistics:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())

# Log-transform Frequency/Monetary (heavily right-skewed) before scaling
rfm_log = rfm.copy()
rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
rfm_log["Monetary"] = np.log1p(rfm_log["Monetary"].clip(lower=0))

X = rfm_log[["Recency", "Frequency", "Monetary"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Select k via silhouette score ---
print("\nSelecting k via silhouette score...")
sil_scores = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores[k] = score
    print(f"  k={k}: silhouette={score:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\nBest k = {best_k} (silhouette={sil_scores[best_k]:.4f})")

# --- Fit final model ---
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, rfm["Cluster"])

cluster_profile = rfm.groupby("Cluster").agg(
    Size=("Customer ID", "count"),
    Recency_mean=("Recency", "mean"),
    Frequency_mean=("Frequency", "mean"),
    Monetary_mean=("Monetary", "mean"),
    Monetary_total=("Monetary", "sum"),
).round(1)
cluster_profile["Size_pct"] = (cluster_profile["Size"] / cluster_profile["Size"].sum() * 100).round(1)
print("\nCluster profile:")
print(cluster_profile)

# --- Plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ks = list(sil_scores.keys())
axes[0].plot(ks, [sil_scores[k] for k in ks], marker="o")
axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.5, label=f"chosen k={best_k}")
axes[0].set_xlabel("k (number of clusters)")
axes[0].set_ylabel("Silhouette score")
axes[0].set_title("Silhouette score vs k")
axes[0].legend()

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=rfm["Cluster"], cmap="tab10", s=8, alpha=0.6)
axes[1].set_xlabel("PCA component 1")
axes[1].set_ylabel("PCA component 2")
axes[1].set_title(f"Customer segments (PCA projection, k={best_k})")
legend1 = axes[1].legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
axes[1].add_artist(legend1)

plt.tight_layout()
plt.savefig("retail_kmeans_results.png", dpi=150)
print("\nSaved plot: retail_kmeans_results.png")

# --- Save results ---
results = {
    "dataset": "Online Retail II",
    "n_customers": int(len(rfm)),
    "n_transactions_after_cleaning": int(len(df)),
    "silhouette_scores_by_k": {str(k): round(v, 4) for k, v in sil_scores.items()},
    "chosen_k": int(best_k),
    "final_silhouette_score": round(float(final_silhouette), 4),
    "cluster_profile": cluster_profile.reset_index().to_dict(orient="records"),
}
with open("retail_kmeans_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results: retail_kmeans_results.json")

rfm.to_csv("retail_rfm_with_clusters.csv", index=False)
print("Saved: retail_rfm_with_clusters.csv")
