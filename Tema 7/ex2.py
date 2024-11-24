from scipy.stats import poisson
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

lambda_ = 4
X = poisson.rvs(lambda_, size=100, random_state=1) 

print("Calls received in the first 10 days:", X[:10])

counts = Counter(X)
values = sorted(counts.keys())  
frequencies = [counts[val] / len(X) for val in values]  

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.bar(values, frequencies, width=0.8, alpha=0.7, color='blue', edgecolor='black')

plt.ylabel("Probability")
plt.xlabel("Number of Calls (k)")
plt.title("Histogram of Calls Received (Poisson Distribution, λ = 4)")

#plt.show()

"""
Point 2
"""


lambda_hat_values = np.linspace(2, 6, 100) 
log_likelihoods = [np.sum(poisson.logpmf(X, mu=lmb)) for lmb in lambda_hat_values]

plt.figure(figsize=(8, 5))
plt.plot(lambda_hat_values, log_likelihoods, label="Log-Likelihood", color="blue")
plt.axvline(x=lambda_, color="red", linestyle="--", label=f"True λ = {lambda_}")
plt.title("Log-Likelihood of Poisson Data")
plt.xlabel(r"$\hat{\lambda}$")
plt.ylabel("Log-Likelihood")
plt.grid(alpha=0.7, linestyle="--")
plt.legend()
plt.show()

"""
Point 3
"""
lambda_hat = np.mean(X)
print(f"MLE for lambda: {lambda_hat:.4f}")

"""
Point 4
"""

lambda_hat_analytical = lambda_hat_values[np.argmax(log_likelihoods)]

print(f"Lambda hat analytical: {lambda_hat_analytical:.4f}") 


