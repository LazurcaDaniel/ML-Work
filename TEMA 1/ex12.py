"""
Exercise 12
S   :  2     3    4     5    6     7    8     9    10    11    12
P(S): 1/36 1/18  1/12  1/9  5/36  1/6  5/36  1/9  1/12  1/18  1/36
"""
import matplotlib.pyplot as plt
import numpy as np

S = [2,3,4,5,6,7,8,9,10,11,12]
P = [1/36,1/18,1/12,1/9,5/36,1/6,5/36,1/9,1/12,1/18,1/36]

"""
Point a
"""
fig, ax = plt.subplots(1,1)
ax.bar(S,P)
plt.xlabel("Sum (S)")
plt.ylabel("Probability")
plt.title("probability distribution of S")
plt.show()

"""
Point b

E[X] = sum(s*P[s]) for s in S (7.0 in this case)
"""

mean = sum(S[i] * P[i] for i in range(len(S)))
"""
Var(S) = sum(P[s]*(s-mean)**2) for s in S (5.833333 in this case)
"""

variation = sum(P[i] * ((S[i] - mean)**2) for i in range(len(S)))
print("E[S] = ",mean)
print("Var[S] = ",variation)