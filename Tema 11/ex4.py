import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import norm

# Dataset
d = pd.DataFrame({
    'X1': np.concatenate([norm.rvs(0, 1, 100, random_state=1), norm.rvs(1, 1, 100, random_state=3)]),
    'X2': np.concatenate([norm.rvs(0, 1, 100, random_state=2), norm.rvs(1, 1, 100, random_state=4)]),
    'Y': [1]*100 + [0]*100
})
X, Y = d[['X1', 'X2']], d['Y']

# Plot the dataset
def plot_dataset():
    plt.figure(figsize=(8, 6))
    plt.scatter(d['X1'][:100], d['X2'][:100], color='red', label='Class 1')
    plt.scatter(d['X1'][100:], d['X2'][100:], color='green', label='Class 0')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Dataset')
    plt.legend()
    plt.grid()
    plt.show()

plot_dataset()

# Evaluate training and CVLOO errors
training_errors = []
cv_errors = []
n_estimators_range = range(1, 16)

# Use StratifiedKFold for proper splitting
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_estimators in n_estimators_range:
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators, algorithm="SAMME")
    ada.fit(X, Y)

    # Training error
    train_error = 1 - ada.score(X, Y)
    training_errors.append(train_error)

    # Cross-validation error
    cv_error = 1 - np.mean(cross_val_score(ada, X, Y, cv=skf))
    cv_errors.append(cv_error)

# Plot training and CV errors
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, training_errors, label='Training Error', marker='o')
plt.plot(n_estimators_range, cv_errors, label='CV Error', marker='o')
plt.xlabel('Number of Weak Learners')
plt.ylabel('Error')
plt.title('Training and CV Errors vs Number of Weak Learners')
plt.legend()
plt.grid()
plt.show()

# Find the optimal number of weak learners
optimal_n_estimators = n_estimators_range[np.argmin(cv_errors)]
print(f'The optimal number of weak learners is: {optimal_n_estimators}')
