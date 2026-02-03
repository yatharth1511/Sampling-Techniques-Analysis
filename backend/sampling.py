import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

# 1. Random Sampling

def random_sampling(X, y, n_samples):
    data = pd.concat([X, y], axis=1)
    return data.sample(n=n_samples, random_state=42)

# 2. Cluster Sampling

def cluster_sampling(X, y, n_clusters=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    data = X.copy()
    data["Class"] = y
    data["Cluster"] = clusters

    sampled = (
        data.groupby("Cluster", group_keys=False)
        .apply(lambda x: x.sample(frac=0.5, random_state=42))
    )

    return sampled.drop("Cluster", axis=1)

# 3. Bootstrap Sampling

def bootstrap_sampling(X, y):
    data = pd.concat([X, y], axis=1)
    boot = resample(data, replace=True, n_samples=len(data), random_state=42)
    return boot

# 4. Stratified Sampling

def stratified_sampling(X, y):
    data = pd.concat([X, y], axis=1)
    sampled = (
        data.groupby("Class", group_keys=False)
        .apply(lambda x: x.sample(frac=0.8, random_state=42))
    )
    return sampled

# 5. Strategic Sampling (distance-based)

def strategic_sampling(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    centroid = X_scaled.mean(axis=0)
    distances = np.linalg.norm(X_scaled - centroid, axis=1)

    data = X.copy()
    data["Class"] = y
    data["distance"] = distances

    strategic = data.nsmallest(int(0.7 * len(data)), "distance")
    return strategic.drop("distance", axis=1)
