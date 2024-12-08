import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from statistics import mean
from sklearn.tree import DecisionTreeClassifier
d = pd.DataFrame({
    'X1': [1, 2, 4, 5, 7],
    'X2': [2, 1, 5, 4, 3],
    'Y': [0, 0, 0, 1, 1]
})
X, Y = d[['X1', 'X2']], d['Y']

c= ['green' if l == 0 else 'red' for l in Y]
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(X['X1'], X['X2'], color=c)
plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X.values, Y)

loo = LeaveOneOut()
scores = cross_val_score(knn, X, Y, cv=loo)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
tree.fit(X, Y)


cvloo_score_tree = cross_val_score(tree, X, Y, cv=loo, scoring='accuracy')

print("Training accuracy for ID3: ", tree.score(X,Y))
print("Training accuracy for 1-NN", knn.score(X, Y))


print(f'CVLOO error for ID3 (Decision Tree): {np.mean(cvloo_score_tree):.4f}')
print("Mean CVLOO score for 1-NN: ", mean(scores))