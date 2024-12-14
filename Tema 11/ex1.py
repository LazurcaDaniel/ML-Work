import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

d = pd.DataFrame({
    'X': [-1, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8],
    'Y': [1, 1, 1, -1, -1, -1, 1]
})
X, Y = d[['X']].values.flatten(), d['Y'].values

# Plot the dataset
plt.scatter(X[Y == 1], [0] * sum(Y == 1), color='blue', label='Y=1', s=100)
plt.scatter(X[Y == -1], [0] * sum(Y == -1), color='red', label='Y=-1', s=100)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('X')
plt.title('Dataset Visualization')
plt.legend()
plt.show()


n = len(Y)
weights = np.ones(n) / n


def find_best_stump(X, Y, weights):
    thresholds = np.sort(np.unique(X) - 1e-8)  # Candidate thresholds
    best_error = float('inf')
    best_stump = None

    for threshold in thresholds:
        for polarity in [1, -1]:  # Test both directions
            predictions = polarity * np.sign(X - threshold)
            error = np.sum(weights[Y != predictions])

            if error < best_error:
                best_error = error
                best_stump = {'threshold': threshold, 'polarity': polarity}

    return best_stump, best_error

stump1, e1 = find_best_stump(X, Y, weights)
threshold1, polarity1 = stump1['threshold'], stump1['polarity']


alpha1 = 0.5 * np.log((1 - e1) / e1)


predictions1 = polarity1 * np.sign(X - threshold1)
weights = weights * np.exp(-alpha1 * Y * predictions1)
weights /= np.sum(weights)  

print(f"First weak learner: Threshold={threshold1}, Polarity={polarity1}")
print(f"Training error (e1) = {e1}")
print(f"Alpha1 = {alpha1}")
print(f"Updated weights = {weights}")


x_vals = np.linspace(-1.2, 1.2, 500)
y_vals = polarity1 * np.sign(x_vals - threshold1)
plt.plot(x_vals, y_vals, label='Decision Boundary (Stump 1)', color='green')
plt.scatter(X[Y == 1], [0] * sum(Y == 1), color='blue', label='Y=1', s=100)
plt.scatter(X[Y == -1], [0] * sum(Y == -1), color='red', label='Y=-1', s=100)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('X')
plt.title('Decision Surface of First Weak Learner')
plt.legend()
plt.show()


stump2, e2 = find_best_stump(X, Y, weights)
threshold2, polarity2 = stump2['threshold'], stump2['polarity']
alpha2 = 0.5 * np.log((1 - e2) / e2)


predictions2 = polarity2 * np.sign(X - threshold2)
ensemble_predictions = alpha1 * predictions1 + alpha2 * predictions2

plt.plot(x_vals, alpha1 * np.sign(x_vals - threshold1), label='Stump 1', color='green')
plt.plot(x_vals, alpha2 * np.sign(x_vals - threshold2), label='Stump 2', color='purple')
plt.scatter(X[Y == 1], [0] * sum(Y == 1), color='blue', label='Y=1', s=100)
plt.scatter(X[Y == -1], [0] * sum(Y == -1), color='red', label='Y=-1', s=100)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('X')
plt.title('Decision Surface After Second Weak Learner')
plt.legend()
plt.show()

print(f"Second weak learner: Threshold={threshold2}, Polarity={polarity2}")
print(f"Training error (e2) = {e2}")
print(f"Alpha2 = {alpha2}")
