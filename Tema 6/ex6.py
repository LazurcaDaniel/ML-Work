import pandas as pd
import numpy as np

# Define feature names
features = [f'w{i}' for i in range(1, 9)]

# Data for politics and sport categories
politics = pd.DataFrame([
    (1, 0, 1, 1, 1, 0, 1, 1),
    (0, 0, 0, 1, 0, 0, 1, 1),
    (1, 0, 0, 1, 1, 0, 1, 0),
    (0, 1, 0, 0, 1, 1, 0, 1),
    (0, 0, 0, 1, 1, 0, 1, 1),
    (0, 0, 0, 1, 1, 0, 0, 1)
], columns=features)

sport = pd.DataFrame([
    (1, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0),
    (1, 1, 0, 1, 0, 0, 0, 0),
    (1, 1, 0, 1, 0, 0, 0, 1),
    (1, 1, 0, 1, 1, 0, 0, 0),
    (0, 0, 0, 1, 0, 1, 0, 0),
    (1, 1, 1, 1, 1, 0, 1, 0)
], columns=features)

# Target document to classify
x = [1, 0, 0, 1, 1, 1, 1, 0]

# Calculate prior probabilities
p_politics = len(politics) / (len(politics) + len(sport))
p_sport = len(sport) / (len(politics) + len(sport))

# Function to calculate conditional probabilities
def calc_conditional_probabilities(df, x):
    probs = []
    for i, feature in enumerate(features):
        prob = df[feature].mean() if x[i] == 1 else 1 - df[feature].mean()
        probs.append(prob)
    return np.prod(probs)

# Calculate P(x|politics) and P(x|sport)
p_x_given_politics = calc_conditional_probabilities(politics, x)
p_x_given_sport = calc_conditional_probabilities(sport, x)

# Calculate unnormalized posteriors
p_politics_given_x = p_x_given_politics * p_politics
p_sport_given_x = p_x_given_sport * p_sport

# Normalize to find final probabilities
total_prob = p_politics_given_x + p_sport_given_x
prob_politics = p_politics_given_x / total_prob
prob_sport = p_sport_given_x / total_prob

# Output the probabilities
print(f"Probabilitatea ca documentul să fie despre 'politics': {prob_politics:.4f}")
print(f"Probabilitatea ca documentul să fie despre 'sport': {prob_sport:.4f}")
