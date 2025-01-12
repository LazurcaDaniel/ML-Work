from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import numpy as np
import pandas as pd

np.random.seed(0)
X1, _ = ds.make_circles(n_samples=1500, factor=.5,  noise=.05)
X2, _ = ds.make_blobs(n_samples=1500,
                      cluster_std=[1.0, 2.5, 0.5],
                      random_state=170)

d1 = pd.DataFrame(X1, columns=['X1', 'X2'])
d2 = pd.DataFrame(X2, columns=['X1', 'X2'])

def plot_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data['X1'], data['X2'], c=labels, cmap='tab10', s=10)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

for linkage in ['single', 'average']:
    # Single Linkage Clustering
    ac_d1 = AgglomerativeClustering(linkage=linkage, n_clusters=2)
    labels_d1 = ac_d1.fit_predict(d1)

    plot_clusters(d1, labels_d1, f"d1: {linkage.capitalize()} Linkage Clustering")

# Clustering and plotting for d2
for linkage in ['single', 'average']:
    # Single Linkage Clustering
    ac_d2 = AgglomerativeClustering(linkage=linkage, n_clusters=2)
    labels_d2 = ac_d2.fit_predict(d2)

    plot_clusters(d2, labels_d2, f"d2: {linkage.capitalize()} Linkage Clustering")