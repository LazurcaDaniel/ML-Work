from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Create datasets
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# Anisotropically distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
anis = np.dot(X, transformation)

# Datasets d1 and d2
d1, d2 = X, anis

# Apply k-means with k=3
kmeans_d1 = KMeans(n_clusters=3, random_state=random_state).fit(d1)
kmeans_d2 = KMeans(n_clusters=3, random_state=random_state).fit(d2)

# Predicted clusters
labels_d1 = kmeans_d1.labels_
labels_d2 = kmeans_d2.labels_

# Plotting function
def plot_clusters(data, labels, centroids, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=30, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x', label='Centroids')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()

# Plot clusters for d1
plot_clusters(d1, labels_d1, kmeans_d1.cluster_centers_, title="Clusters for d1 (Original Data)")

# Plot clusters for d2
plot_clusters(d2, labels_d2, kmeans_d2.cluster_centers_, title="Clusters for d2 (Anisotropic Data)")

"""
The clusters in d1 look more "natural" because the data matches the assumptions of k-means (spherical, evenly spaced clusters).
"""