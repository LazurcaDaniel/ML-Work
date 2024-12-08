import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

d = pd.DataFrame({
    'X1' : [-1, 0 , 1],
    'X2' : [0 , 1, 0 ],
    'Y' : ['R', 'G', 'B']
})

X, Y = d[['X1','X2']], d['Y']
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, Y)

x_min, x_max = X['X1'].min() - 1, X['X1'].max() + 1
y_min, y_max = X['X2'].min() - 1, X['X2'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))


Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

label_to_color_idx = {'R': 0, 'G': 1, 'B': 2}
Z = np.vectorize(label_to_color_idx.get)(Z)

fig, ax = plt.subplots(figsize=(6, 6))
color_map = ['red', 'green', 'blue']
plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], colors=color_map)

# Plot the original points
c = [color_map[label_to_color_idx[label]] for label in Y]
plt.scatter(X['X1'], X['X2'], color=c, edgecolor='k', s=100)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('1-NN Decision Surface')
plt.show()
