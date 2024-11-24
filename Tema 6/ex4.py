import numpy as np
import pandas as pd
from scipy.stats import bernoulli, pearsonr
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Function to create correlated features and target variable
num_samples = 10000

def generate_data_with_correlation(correlation):
    feature1 = bernoulli.rvs(0.5, size=num_samples, random_state=1)
    df = pd.DataFrame({'feature1': feature1})
    
    correlation_mask = bernoulli.rvs(correlation, size=num_samples, random_state=2)
    random_bits = bernoulli.rvs(0.5, size=num_samples, random_state=3)
    df['feature2'] = df['feature1'] & correlation_mask | random_bits & ~correlation_mask
    df['target'] = df['feature1'] & ~df['feature2']
    return df

# Define correlation levels and store error values
correlation_levels = np.linspace(0, 1, 20)  # From 0 to 1 in 20 steps
naive_bayes_errors = []
decision_tree_errors = []

# Iterate through each correlation level
for correlation in correlation_levels:
    # Generate data for the given correlation level
    data = generate_data_with_correlation(correlation)
    X = data[['feature1', 'feature2']]
    y = data['target']
    
    # Naive Bayes Classifier
    nb_classifier = BernoulliNB()
    nb_classifier.fit(X, y)
    nb_predictions = nb_classifier.predict(X)
    nb_error_rate = 1 - accuracy_score(y, nb_predictions)
    naive_bayes_errors.append(nb_error_rate)
    
    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X, y)
    dt_predictions = dt_classifier.predict(X)
    dt_error_rate = 1 - accuracy_score(y, dt_predictions)
    decision_tree_errors.append(dt_error_rate)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(correlation_levels, naive_bayes_errors, label='Naive Bayes Error', marker='o', linestyle='--')
plt.plot(correlation_levels, decision_tree_errors, label='Decision Tree Error', marker='x', linestyle='-')
plt.title('Training Error vs. Feature Correlation')
plt.xlabel('Feature Correlation')
plt.ylabel('Training Error')
plt.legend()
plt.grid(True)
plt.show()
