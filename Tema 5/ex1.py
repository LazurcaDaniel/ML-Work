import pandas as pd
from sklearn.naive_bayes import BernoulliNB

input_attr = ['A','B','C']
output_attr = 'Y'

d = pd.DataFrame({'A': [0, 0, 1, 0, 1, 1, 1],
                  'B': [0, 1, 1, 0, 1, 0, 1],
                  'C': [1, 0, 0, 1, 1, 0, 0],
                  'Y': [0, 0, 0, 1, 1, 1, 1]})
#Point 1
test = pd.DataFrame(
  [(0, 0, 1)],columns = ['A','B','C'])

X = d[input_attr]
Y = d[output_attr]
cl = cl = BernoulliNB(alpha=1e-10).fit(X, Y)
print(cl.classes_)
print(cl.predict_proba(test))
#With probability of 52% for value 1, Naive Bayes will classify (0,0,1) as 1

#Point 2

P_Y_0 = (Y == 0).sum() / len(Y)  # P(Y = 0)
P_Y_1 = (Y == 1).sum() / len(Y)  # P(Y = 1)
print(f'P(Y = 0) = {P_Y_0}')
print(f'P(Y = 1) = {P_Y_1}')

# Calculate P[ai | Y = 0] 
P_A_0_given_Y_0 = ((X.loc[Y == 0, 'A'] == 0).sum()) / (Y == 0).sum()
P_B_0_given_Y_0 = ((X.loc[Y == 0, 'B'] == 0).sum()) / (Y == 0).sum()
P_C_1_given_Y_0 = ((X.loc[Y == 0, 'C'] == 1).sum()) / (Y == 0).sum()
print(f'P(A = 0 | Y = 0) = {P_A_0_given_Y_0}')
print(f'P(B = 0 | Y = 0) = {P_B_0_given_Y_0}')
print(f'P(C = 1 | Y = 0) = {P_C_1_given_Y_0}')

# Calculate P[ai | Y = 1]
P_A_0_given_Y_1 = ((X.loc[Y == 1, 'A'] == 0).sum()) / (Y == 1).sum()
P_B_0_given_Y_1 = ((X.loc[Y == 1, 'B'] == 0).sum()) / (Y == 1).sum()
P_C_1_given_Y_1 = ((X.loc[Y == 1, 'C'] == 1).sum()) / (Y == 1).sum()
print(f'P(A = 0 | Y = 1) = {P_A_0_given_Y_1}')
print(f'P(B = 0 | Y = 1) = {P_B_0_given_Y_1}')
print(f'P(C = 1 | Y = 1) = {P_C_1_given_Y_1}')

# Calculate posterior probabilities
P_Y_0_given_test = P_Y_0 * P_A_0_given_Y_0 * P_B_0_given_Y_0 * P_C_1_given_Y_0
P_Y_1_given_test = P_Y_1 * P_A_0_given_Y_1 * P_B_0_given_Y_1 * P_C_1_given_Y_1
print(f'P(Y = 0 | A=0,B=0,C=1) = {P_Y_0_given_test}')
print(f'P(Y = 1 | A=0,B=0,C=1) = {P_Y_1_given_test}')

total = P_Y_0_given_test + P_Y_1_given_test
P_Y_0_given_test /= total
P_Y_1_given_test /= total


print("\nManual calculation results:")
print(f'Total for test = {total}')
print(f"P(Y = 0 | A = 0, B = 0, C = 1) = {P_Y_0_given_test:.4f}")
print(f"P(Y = 1 | A = 0, B = 0, C = 1) = {P_Y_1_given_test:.4f}")