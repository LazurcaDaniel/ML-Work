import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import pairwise_distances_argmin
import pandas as pd

# Data
data = {
    'A': [2, 10], 'B': [2, 5], 'C': [8, 4], 'D': [5, 8], 
    'E': [7, 5], 'F': [6, 4], 'G': [1, 2], 'H': [4, 9]
}
d = pd.DataFrame.from_dict(data, orient='index', columns=['X', 'Y'])

# Initial centroids (A, D, G)
initial_centroids = d.loc[['A', 'D', 'G']].values

# Function to plot clusters and Voronoi diagram
def plot_kmeans_iteration(points, centroids, labels, iteration):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    
    # Plot points with cluster colors
    for cluster_idx in range(len(centroids)):
        cluster_points = points[labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[cluster_idx], label=f'Cluster {cluster_idx+1}')
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x', s=150, label='Centroids')
    
    # Voronoi diagram
    vor = Voronoi(centroids)
    voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='orange', line_width=2)
    
    # Labels and title
    plt.title(f'K-Means Iteration {iteration}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.show()

# K-means loop
points = d.values
centroids = initial_centroids
iteration = 0

while True:
    # Assign points to nearest centroid
    labels = pairwise_distances_argmin(points, centroids)
    
    # Plot current iteration
    plot_kmeans_iteration(points, centroids, labels, iteration)
    
    # Update centroids
    new_centroids = np.array([points[labels == i].mean(axis=0) for i in range(len(centroids))])
    
    # Check for convergence
    if np.all(centroids == new_centroids):
        break
    
    centroids = new_centroids
    iteration += 1
