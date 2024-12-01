import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


X, y = make_moons(n_samples=200, noise=0.2, random_state=42)


def plot_decision_surface(X, y, model, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    ax.set_title(title)


log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
log_reg.fit(X, y)

loo = LeaveOneOut()
cvloo_score_log_reg = cross_val_score(log_reg, X, y, cv=loo, scoring='accuracy')
cvloo_error_log_reg = 1 - np.mean(cvloo_score_log_reg)
print(f'CVLOO error for Logistic Regression: {cvloo_error_log_reg:.4f}')


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_decision_surface(X, y, log_reg, plt.gca(), 'Logistic Regression')

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
tree.fit(X, y)


cvloo_score_tree = cross_val_score(tree, X, y, cv=loo, scoring='accuracy')
cvloo_error_tree = 1 - np.mean(cvloo_score_tree)
print(f'CVLOO error for ID3 (Decision Tree): {cvloo_error_tree:.4f}')


plt.subplot(1, 2, 2)
plot_decision_surface(X, y, tree, plt.gca(), 'ID3 (Decision Tree)')
plt.tight_layout()
plt.show()

