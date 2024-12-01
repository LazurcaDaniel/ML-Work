import numpy as np
from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=200, cluster_std=3, centers=2, random_state=42)


def add_intercept(X):
    """Add 1 as the first column of X"""
    return np.hstack((np.ones((len(X), 1)), X))

X = add_intercept(X)


w = np.zeros(X.shape[1])

eta = 0.01


steps = 10


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for step in range(steps):
    gradient = np.zeros(X.shape[1])  
    
    for i in range(X.shape[0]):
        xi = X[i] 
        yi = y[i]  
        prediction = sigmoid(np.dot(w, xi)) 
        gradient += (yi - prediction) * xi  


    w += eta * gradient

    print(f"Step {step + 1}: w = {w}")


print(f"Final weights after {steps} steps: w = {w}")
