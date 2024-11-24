import pandas as pd
import numpy as np
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

d = pd.DataFrame({'X1': [0, 0, 1, 1, 0, 0, 1, 1],
                  'X2': [0, 0, 0, 0, 1, 1, 1, 1],
                  'C' : [2, 18, 4, 1, 4, 1, 2, 18],
                  'Y' : [0, 1, 0, 1, 0, 1, 0, 1]})
d=apply_counts(d, 'C')

print(d)
# class BernoulliJB:
#     def __init__(self):
#         self.class_prob_ = None  # Stores P(Class)
#         self.joint_prob_ = None  # Stores joint P(Feature1, Feature2, ..., FeatureN | Class)
#         self.classes_ = None  # Stores unique classes
    
#     def fit(self, X, y):
#         self.classes_, class_counts = np.unique(y, return_counts=True)
#         n_classes = len(self.classes_)
#         n_features = X.shape[1]
        

#         self.class_prob_ = class_counts / len(y)
        

#         self.joint_prob_ = np.zeros((n_classes, 2**n_features))
        

#         for idx, class_label in enumerate(self.classes_):
#             X_class = X[y == class_label]
#             n_samples_class = X_class.shape[0]
            

#             for i in range(2**n_features):

#                 binary_repr = np.array([int(b) for b in np.binary_repr(i, width=n_features)])
                

#                 prob = np.mean(np.all(X_class == binary_repr, axis=1))
#                 self.joint_prob_[idx, i] = prob
                
#     def predict_proba(self, X):
#         n_samples = X.shape[0]
#         n_features = X.shape[1]
#         n_classes = len(self.classes_)
        
#         prob_X = np.zeros((n_samples, n_classes))
        
#         for i in range(n_samples):
#             sample = X[i]
#             index = int("".join(sample.astype(int).astype(str)), 2)
            
#             joint_probs = self.joint_prob_[:, index] * self.class_prob_
#             prob_X[i, :] = joint_probs / joint_probs.sum()  
        
#         return prob_X

# model = BernoulliJB()
# X = d[['X1','X2']].values
# Y = d['Y'].values
# model.fit(X,Y)


# num_classes = len(model.class_prob_)


# num_features = model.joint_prob_.shape[1]


# total_probabilities = num_classes * (2 ** num_features)

# print("Number of classes:", num_classes)
# print("Number of features:", num_features)
# print("Total probabilities estimated:", total_probabilities)

# X_test = np.array([[0, 0]])  
# probs = model.predict_proba(X_test)

# print("Probability estimates for BernoulliJB:", probs)

# naive_model = BernoulliNB(alpha = 0)  
# naive_model.fit(X, Y)


# naive_probs = naive_model.predict_proba(X_test)

# print("Predicted naive probabilities:", naive_probs)