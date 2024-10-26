import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from statistics import mean

d = pd.DataFrame({'X1': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
                  'X2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                  'Y' : [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]})

Y = d['Y']
X = d[['X1','X2']]
classifier = tree.DecisionTreeClassifier(criterion='entropy')
dt = classifier.fit(X,Y)
fig, ax = plt.subplots(figsize=(7, 8))
f = tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=X.columns)
plt.show()
loo = LeaveOneOut()

underfit_classifier =  DummyClassifier(strategy="most_frequent")
underfit_dt = underfit_classifier.fit(X,Y)

print("Mean CVLOO accuracy for OVERFIT: ", mean(cross_val_score(classifier, X, Y, cv=loo)))
print("Mean CVLOO accuracy for UNDERFIT: ", underfit_dt.score(X,Y))