"""
Week 3 - Project 1: Customer Segmentation (K-Means Clustering)
------------------------------------------------------------------
Steps:
  1. Load and explore the dataset
  2. Clean & scale features
  3. Use the Elbow Method to choose k
  4. Apply K-Means and visualize clusters
  5. Profile clusters (age, spending score) to derive marketing actions
  6. Save cluster labels and generate a short report per segment

Dataset: Mall Customers dataset (Kaggle: "Mall Customer Segmentation Data").
Expected columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------
# Step 1 — Load and explore the dataset
# -----------------------------------------------------------------------
DATA_PATH = "Mall_Customers.csv"  

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nHead:\n", df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary stats:\n", df.describe())

# -----------------------------------------------------------------------
# Step 2 — Clean & scale features
# -----------------------------------------------------------------------
# Drop rows with missing values in the columns we need
df = df.dropna(subset=["Age", "Annual Income (k$)", "Spending Score (1-100)"])

# Features used for clustering. Age, income, and spending score together
# give K-Means a well-rounded view of customer behaviour.
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------------
# Step 3 — Elbow Method to choose k
# -----------------------------------------------------------------------
inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.show()

# Inspect the plot: k is chosen where the inertia curve visibly "bends"
# (the elbow). For the Mall Customers dataset this is typically k=5.
OPTIMAL_K = 5  # <-- update this after inspecting your own elbow plot

# -----------------------------------------------------------------------
# Step 4 — Apply K-Means and visualize clusters
# -----------------------------------------------------------------------
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    s=80,
)
plt.title(f"Customer Segments (k={OPTIMAL_K})")
plt.tight_layout()
plt.savefig("customer_clusters.png")
plt.show()

# -----------------------------------------------------------------------
# Step 5 — Profile clusters to derive marketing actions
# -----------------------------------------------------------------------
cluster_profile = df.groupby("Cluster")[features].mean().round(1)
cluster_profile["Count"] = df["Cluster"].value_counts().sort_index()
print("\nCluster Profile:\n", cluster_profile)

# Example interpretation template (edit based on your actual cluster
# profile numbers):
#   - High income, high spending score -> "Premium" customers: target with
#     loyalty programs and premium product launches.
#   - High income, low spending score -> "Cautious spenders": target with
#     personalized offers to increase engagement.
#   - Low income, high spending score -> "Impulse buyers": target with
#     budget-friendly bundles and installment options.
#   - Low income, low spending score -> "Price-sensitive": target with
#     discounts and value-focused promotions.
#   - Mid income, mid spending score -> "Average" customers: general
#     seasonal campaigns.

# -----------------------------------------------------------------------
# Step 6 — Save cluster labels and a short report per segment
# -----------------------------------------------------------------------
df.to_csv("customers_with_clusters.csv", index=False)
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "customer_scaler.pkl")

with open("cluster_report.txt", "w") as f:
    f.write("Customer Segmentation Report\n")
    f.write("=" * 40 + "\n\n")
    for cluster_id, row in cluster_profile.iterrows():
        f.write(f"Cluster {cluster_id} (n={int(row['Count'])})\n")
        f.write(f"  Avg Age:            {row['Age']}\n")
        f.write(f"  Avg Annual Income:  {row['Annual Income (k$)']} k$\n")
        f.write(f"  Avg Spending Score: {row['Spending Score (1-100)']}\n\n")

print("\nSaved: customers_with_clusters.csv, kmeans_model.pkl, "
      "customer_scaler.pkl, cluster_report.txt")
