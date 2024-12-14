import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# Create the dataset
x_red = norm.rvs(0, 1, 100, random_state=1)
y_red = norm.rvs(0, 1, 100, random_state=2)
x_green = norm.rvs(1, 1, 100, random_state=3)
y_green = norm.rvs(1, 1, 100, random_state=4)
d = pd.DataFrame({
    'X1': np.concatenate([x_red, x_green]),
    'X2': np.concatenate([y_red, y_green]),
    'Y': [1] * 100 + [0] * 100
})
X, Y = d[['X1', 'X2']].values, d['Y'].values

# Plot the dataset
plt.scatter(d[d['Y'] == 1]['X1'], d[d['Y'] == 1]['X2'], color='green', label='Class 1', s=30)
plt.scatter(d[d['Y'] == 0]['X1'], d[d['Y'] == 0]['X2'], color='red', label='Class 0', s=30)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Dataset Visualization')
plt.legend()
plt.show()

# Initialize weights equally
n = len(Y)
weights = np.ones(n) / n

# Decision stump (weak learner) implementation
def find_best_stump_2D(X, Y, weights):
    n_features = X.shape[1]
    best_error = float('inf')
    best_stump = None

    for feature in range(n_features):
        unique_values = np.sort(np.unique(X[:, feature]))
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2 
        for threshold in thresholds:
            for polarity in [1, -1]:
                predictions = polarity * np.sign(X[:, feature] - threshold)
                error = np.sum(weights[Y != predictions])

                if error < best_error:
                    best_error = error
                    best_stump = {
                        'feature': feature,
                        'threshold': threshold,
                        'polarity': polarity
                    }

    return best_stump, best_error

# Fit AdaBoost
num_learners = 10
alphas = []
stumps = []
for t in range(num_learners):
    stump, error = find_best_stump_2D(X, Y, weights)
    alpha = 0.5 * np.log((1 - error) / error)

    # Update weights
    predictions = stump['polarity'] * np.sign(X[:, stump['feature']] - stump['threshold'])
    weights = weights * np.exp(-alpha * Y * predictions)
    weights /= np.sum(weights)

    stumps.append(stump)
    alphas.append(alpha)

# Compute training error for AdaBoost
ada_predictions = np.zeros(n)
for alpha, stump in zip(alphas, stumps):
    ada_predictions += alpha * (stump['polarity'] * np.sign(X[:, stump['feature']] - stump['threshold']))
ada_predictions = np.sign(ada_predictions)
ada_training_error = np.mean(ada_predictions != Y)

# Implement ID3 for comparison (simplified)
def id3_train(X, Y):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=1)  # Decision stumps
    clf.fit(X, Y)
    return clf

def id3_training_error(X, Y, clf):
    predictions = clf.predict(X)
    return np.mean(predictions != Y)

id3_clf = id3_train(X, Y)
id3_training_error_value = id3_training_error(X, Y, id3_clf)

# Compare training errors
print(f"AdaBoost Training Error: {ada_training_error}")
print(f"ID3 Training Error: {id3_training_error_value}")

# Compute CVLOO error (Leave-One-Out Cross-Validation)
def cvloo_error_ada(X, Y):
    n = len(Y)
    errors = []
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i)
        X_test = X[i].reshape(1, -1)
        Y_test = Y[i]

        # Fit AdaBoost on remaining data
        weights = np.ones(len(Y_train)) / len(Y_train)
        alphas = []
        stumps = []
        for t in range(num_learners):
            stump, error = find_best_stump_2D(X_train, Y_train, weights)
            alpha = 0.5 * np.log((1 - error) / error)

            predictions = stump['polarity'] * np.sign(X_train[:, stump['feature']] - stump['threshold'])
            weights = weights * np.exp(-alpha * Y_train * predictions)
            weights /= np.sum(weights)

            stumps.append(stump)
            alphas.append(alpha)

        # Make prediction on left-out instance
        ada_prediction = 0
        for alpha, stump in zip(alphas, stumps):
            ada_prediction += alpha * (stump['polarity'] * np.sign(X_test[0, stump['feature']] - stump['threshold']))
        ada_prediction = np.sign(ada_prediction)

        errors.append(ada_prediction != Y_test)

    return np.mean(errors)

def cvloo_error_id3(X, Y):
    from sklearn.tree import DecisionTreeClassifier
    n = len(Y)
    errors = []
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i)
        X_test = X[i].reshape(1, -1)
        Y_test = Y[i]

        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X_train, Y_train)
        errors.append(clf.predict(X_test) != Y_test)

    return np.mean(errors)

ada_cvloo_error = cvloo_error_ada(X, Y)
id3_cvloo_error = cvloo_error_id3(X, Y)

# Compare CVLOO errors
print(f"AdaBoost CVLOO Error: {ada_cvloo_error}")
print(f"ID3 CVLOO Error: {id3_cvloo_error}")
