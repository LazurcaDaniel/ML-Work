import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Dataset
np.random.seed(42)
d = pd.DataFrame({
    'X1': [1, 2, 2.75, 3.25, 4, 5],
    'X2': [1, 2, 1.25, 2.75, 2.25, 3.5],
    'Y': [1, 1, -1, 1, -1, -1]
})
X, Y = d[['X1', 'X2']], d['Y']

# Plot the dataset
def plot_dataset():
    plt.figure(figsize=(8, 6))
    for label, color in zip([1, -1], ['blue', 'red']):
        subset = d[d['Y'] == label]
        plt.scatter(subset['X1'], subset['X2'], label=f'Class {label}', color=color)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Dataset')
    plt.legend()
    plt.grid()
    plt.show()

plot_dataset()

# Train AdaBoost with decision stumps as weak learners
base_learner = DecisionTreeClassifier(max_depth=1)  # Decision stump
ada = AdaBoostClassifier(estimator=base_learner, n_estimators=1, algorithm="SAMME")
ada.fit(X, Y)

# First weak learner
def plot_decision_surface(clf, X, Y, title):
    plt.figure(figsize=(8, 6))
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()

plot_decision_surface(ada.estimators_[0], X, Y, 'First Weak Learner Decision Surface')

# Calculate thresholds at the midpoints between points with different classes
def calculate_thresholds(X, Y, feature):
    sorted_data = X[[feature]].join(Y).sort_values(by=feature)
    thresholds = []
    for i in range(len(sorted_data) - 1):
        if sorted_data['Y'].iloc[i] != sorted_data['Y'].iloc[i + 1]:
            midpoint = (sorted_data[feature].iloc[i] + sorted_data[feature].iloc[i + 1]) / 2
            thresholds.append(midpoint)
    return thresholds

# Plot decision surfaces for all possible weak learners and calculate their error rates
weak_learners = []
errors = []

for feature in ['X1', 'X2']:
    thresholds = calculate_thresholds(X, Y, feature)
    for threshold in thresholds:
        stump = DecisionTreeClassifier(max_depth=1)
        X_temp = X.copy()
        X_temp['threshold'] = X_temp[feature] > threshold
        stump.fit(X_temp[[feature]], Y)
        predictions = stump.predict(X[[feature]])
        error = np.mean(predictions != Y)
        weak_learners.append((stump, feature, threshold))
        errors.append(error)

# Plot each weak learner's decision surface vertically
for i, (stump, feature, threshold) in enumerate(weak_learners):
    x_min, x_max = X[feature].min() - 1, X[feature].max() + 1
    xx = np.arange(x_min, x_max, 0.01)

    plt.figure(figsize=(8, 6))
    plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold {threshold:.2f}')
    plt.scatter(X[feature], Y, c=Y, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.xlim(x_min, x_max)
    plt.ylim(-1.5, 1.5)
    plt.xlabel(feature)
    plt.title(f'Weak Learner {i+1}\nError: {errors[i]:.2f}')
    plt.legend()
    plt.show()
