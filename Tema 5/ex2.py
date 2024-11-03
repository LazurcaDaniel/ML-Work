import pandas as pd 
from sklearn.naive_bayes import BernoulliNB

def apply_counts(df: pd.DataFrame, count_col: str):
    """ Denormalise a dataframe with a 'Counts' column by
    multiplying that column by the count and dropping the 
    count_col. """
    feats = [c for c in df.columns if c != count_col]
    return pd.concat([
        pd.DataFrame([list(r[feats])] * r[count_col], columns=feats)
        for i, r in df.iterrows()
    ], ignore_index=True)

input_attr = ['X1','X2']
target = 'Y'

d_grouped = pd.DataFrame({
    'X1': [0, 0, 1, 1, 0, 0, 1, 1],
    'X2': [0, 0, 0, 0, 1, 1, 1, 1],
    'C' : [2, 18, 4, 1, 4, 1, 2, 18],
    'Y' : [0, 1, 0, 1, 0, 1, 0, 1]})

d = apply_counts(d_grouped, 'C')

"""
Point 1:
It will classify the example as a 1
cl.predict: 1

Point 2:
cl.classes_: [0 1]
cl.predict_proba: [[0.24 0.76]]
"""
test = pd.DataFrame(
  [(0, 0)],columns = input_attr)
X = d[input_attr]
Y = d[target]
cl = cl = BernoulliNB(alpha=1e-10).fit(X, Y)
print(cl.classes_)
print(cl.predict(test))
print(cl.predict_proba(test))


"""Point 3"""
print(cl.class_log_prior_)
print("Number of classes (prior probabilities):", len(cl.class_log_prior_))

print(cl.feature_log_prob_)
print("Number of of features (input variables) used in the model", cl.feature_log_prob_.shape[1])

"""
Point 4
k = number of unique classes of variable Y
n = number of features

The Naive Bayes algorithm would compute the probabilities for each of the k classes of variable Y 
and also, for each of the n features, another k probabilities, meaning that it would calculate n*k probabilities

So in total, for n features, it would calculate:
    k + (n*k) 
probabilities
"""

