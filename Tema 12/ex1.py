from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_dendrogram(model, **kwargs):
    # Create counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Create linkage matrix for dendrogram
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)


# Define dataset
d = pd.DataFrame({
    'X': [-4, -3, -2, -1, 1, 1, 2, 3, 3, 4],
    'Y': [-2, -2, -2, -2, -1, 1, 3, 2, 4, 3]
})

# Function to plot clusters
def plot_clusters(data, labels, title):
    plt.scatter(data['X'], data['Y'], c=labels, cmap='tab10')

    # Find unique clusters
    unique_labels = np.unique(labels)

    for cluster in unique_labels:
        # Get points in the cluster
        cluster_points = data[labels == cluster]
        
        # Compute cluster center (mean of X and Y)
        center_x = cluster_points['X'].mean()
        center_y = cluster_points['Y'].mean()
        
        # Compute radius as the maximum distance from the center
        radius = np.sqrt(((cluster_points['X'] - center_x)**2 + (cluster_points['Y'] - center_y)**2).max())
        
        # Draw a circle
        circle = plt.Circle((center_x, center_y), radius, color='black', fill=False, linestyle='--', linewidth=1.5)
        plt.gca().add_artist(circle)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()

# Clustering with single-linkage
ac_single = AgglomerativeClustering(linkage='single', distance_threshold=0, n_clusters=None)
ac_single = ac_single.fit(d)

# Plot dendrogram for single-linkage
plt.figure(figsize=(10, 6))
plot_dendrogram(ac_single, truncate_mode='level', p=4)
plt.title("Dendrogram (Single Linkage)")
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Extract clusters from single-linkage dendrogram
ac_single_clusters = AgglomerativeClustering(linkage='single', n_clusters=4)
labels_single = ac_single_clusters.fit_predict(d)
plot_clusters(d, labels_single, "Single-Linkage Clustering (Based on Dendrogram)")

# Clustering with complete-linkage
ac_complete = AgglomerativeClustering(linkage='complete', distance_threshold=0, n_clusters=None)
ac_complete = ac_complete.fit(d)

# Plot dendrogram for complete-linkage
plt.figure(figsize=(10, 6))
plot_dendrogram(ac_complete, truncate_mode='level', p=4)
plt.title("Dendrogram (Complete Linkage)")
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Extract clusters from complete-linkage dendrogram
ac_complete_clusters = AgglomerativeClustering(linkage='complete', n_clusters=4)
labels_complete = ac_complete_clusters.fit_predict(d)
plot_clusters(d, labels_complete, "Complete-Linkage Clustering (Based on Dendrogram)")

"""
Differences:
Behaviour:
    Sinkle-Linkage: 
        Single-linkage tends to form long, "chain-like" clusters.
        It is susceptible to noise and outliers because clusters can merge based on just one pair of nearby points.
        It does not impose a preference for compact or spherical clusters.
    Complete Linkage:
        Complete-linkage prefers compact and spherical clusters.
        It minimizes the maximum distance within clusters, resulting in tighter, more compact clusters.
Shape:
    Single-Linkage:	    Irregular/Elongated	
    Complete-Linkage:	Compact/Spherical

"""