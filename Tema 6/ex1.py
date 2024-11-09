import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import numpy as np

# Create the training set
features = ['study', 'free', 'money']
target = 'is_spam'
messages = pd.DataFrame(
[(1, 0, 0, 0),
(0, 0, 1, 0),
(1, 0, 0, 0),
(1, 1, 0, 0)] +
[(0, 1, 0, 1)] * 4 +
[(0, 1, 1, 1)] * 4,
columns=features+[target])
# Create the prediction set
X = messages[features]
y = messages[target]
cl = BernoulliNB(alpha=1e-10).fit(X, y)

def class_log_prior(df: pd.DataFrame):
    total_messages = len(df)
    class_counts = df['is_spam'].value_counts()
    probability = class_counts / total_messages
    return np.log(probability)

def feature_log_prob(df: pd.DataFrame):
    spam = df[df['is_spam'] == 1]
    not_spam = df[df['is_spam'] == 0]

    total_spam = len(spam)
    total_not_spam = len(not_spam)

    feature_probs_class_0 = np.log((not_spam[features].sum() + 1e-10) / total_not_spam)
    feature_probs_class_1 = np.log((spam[features].sum() + 1e-10) / total_spam)

    return np.array([feature_probs_class_0,feature_probs_class_1])

print(cl.classes_)
print(cl.class_log_prior_)
print(class_log_prior(messages))
print(feature_log_prob(messages))
print(cl.feature_log_prob_)




