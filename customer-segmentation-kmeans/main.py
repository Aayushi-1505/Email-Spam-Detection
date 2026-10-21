"""
Customer Segmentation using K-Means Clustering
Dataset: Mall Customers (Kaggle)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.makedirs("images", exist_ok=True)

# 1. Load data
df = pd.read_csv("data/Mall_Customers.csv")
print(df.head())
print(df.info())
print("Missing values:\n", df.isnull().sum())

# 2. Quick EDA
sns.pairplot(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
plt.savefig("images/eda_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Select features and scale them
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

# 4. Elbow method + silhouette score to choose k
inertia, sil = [], []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, labels))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(K, inertia, marker="o")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("Number of clusters (k)")
ax[0].set_ylabel("Inertia")
ax[1].plot(K, sil, marker="o", color="green")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("Number of clusters (k)")
ax[1].set_ylabel("Score")
plt.savefig("images/elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()

best_k = 5  # elbow + silhouette usually point to 5 for this dataset
print(f"Chosen k = {best_k}")

# 5. Final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
print("Silhouette score:", round(silhouette_score(X_scaled, df["Cluster"]), 3))

# 6. Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x=features[0], y=features[1],
                hue="Cluster", palette="tab10", s=70)
centers = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=250, marker="X", label="Centroids")
plt.title("Customer Segments")
plt.legend()
plt.savefig("images/clusters.png", dpi=150, bbox_inches="tight")
plt.close()

# 7. Cluster summary
summary = df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)
summary["Customers"] = df["Cluster"].value_counts().sort_index()
print(summary)

df.to_csv("data/customers_with_clusters.csv", index=False)
print("Done. Charts saved in images/")
