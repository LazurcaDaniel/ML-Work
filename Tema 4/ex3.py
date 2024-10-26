import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

d = pd.DataFrame({'X1': [1, 2, 3, 3, 3, 4, 5, 5, 5],
                  'X2': [2, 3, 1, 2, 4, 4, 1, 2, 4],
                  'Y':  [1, 1, 0, 0, 0, 0, 1, 1, 0]})

Y = d['Y']
X = d[['X1','X2']]
c = ['green' if l == 1 else 'red' for l in Y]
fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(X['X1'],X['X2'], color = c)
classifier = tree.DecisionTreeClassifier(criterion='entropy').fit(X,Y)
DecisionBoundaryDisplay.from_estimator(classifier, X, response_method="predict", ax=ax, alpha=0.3)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Surface with Decision Tree')
plt.show()


X1 = 10 * np.random.random_sample(1000)
X2 = 10 * np.random.random_sample(1000)

Y = (X2 > X1).astype(int) 
c = ['green' if l == 1 else 'red' for l in Y]
d = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
fig1, ax1 = plt.subplots(figsize=(5,5))
plt.scatter(X1,X2, color = c)
X = d[['X1','X2']]
classifier = tree.DecisionTreeClassifier(criterion='entropy').fit(X,Y)
DecisionBoundaryDisplay.from_estimator(classifier, X, response_method="predict", ax=ax1, alpha=0.3)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Surface with Decision Tree')
plt.show()

