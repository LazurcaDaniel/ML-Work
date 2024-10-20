import pandas as pd
import math
titanic = pd.DataFrame([
  ('Upper', 'Male', 'Child', 5, 5),
  ('Upper', 'Male', 'Adult', 175, 57),
  ('Upper', 'Female', 'Child', 1, 1),
  ('Upper', 'Female', 'Adult', 144, 140),
  ('Lower', 'Male', 'Child', 59, 24),
  ('Lower', 'Male', 'Adult', 1492, 281),
  ('Lower', 'Female', 'Child', 44, 27),
  ('Lower', 'Female', 'Adult', 281, 176)
],
columns=['Class', 'Gender', 'Age', 'Passengers', 'Survivors'])

def get_overall_entropy(titanic):
    nr_passengers = sum(titanic['Passengers'])
    nr_survivors = sum(titanic['Survivors'])
    nr_deaths = nr_passengers - nr_survivors
    P = [nr_deaths/nr_passengers, nr_survivors/nr_passengers]
    return P[0]*math.log2(1/P[0]) + P[1] * math.log2(1/P[1])

def get_attribute_ig(titanic,titanic_entropy, attribute):
    values = {}
    for item in titanic[attribute]:
        if item not in values:
            values[item] = 0
        values[item] += 1
    probs_attribute = []
    entropies = []
    for (key,value) in values.items():
        nr_pas = sum(titanic[titanic[attribute] == key]['Passengers'])
        probs_attribute.append(nr_pas/sum(titanic['Passengers']))
        nr_suv = sum(titanic[titanic[attribute] == key]['Survivors'])
        nr_deaths = nr_pas - nr_suv
        
        P = [nr_deaths/nr_pas, nr_suv/nr_pas]
        entropies.append(P[0]*math.log2(1/P[0]) + P[1] * math.log2(1/P[1]))
    entropy_attribute = sum(p*H for p,H in zip(probs_attribute,entropies))
    return titanic_entropy - entropy_attribute

titanic_entropy = get_overall_entropy(titanic)
print(f'Class: {get_attribute_ig(titanic,titanic_entropy,'Class')}')
print(f'Gender: {get_attribute_ig(titanic,titanic_entropy,'Gender')}')
print(f'Age: {get_attribute_ig(titanic,titanic_entropy,'Age')}')
"""Since IG(Gender) is the biggest, we will pick Gender as the root node"""
    
def get_majority_class(titanic, group_by_attribute):
    majority_class = {}
    for group in titanic[group_by_attribute].unique():
        subset = titanic[titanic[group_by_attribute] == group]
        total_passengers = sum(subset['Passengers'])
        total_survivors = sum(subset['Survivors'])
        total_deaths = total_passengers - total_survivors
        if total_survivors > total_deaths:
            majority_class[group] = 'Survive'
        else:
            majority_class[group] = 'Die'
    
    return majority_class

# Step 2: Calculate the accuracy based on the majority class
def calculate_accuracy(titanic, group_by_attribute, majority_class):
    correct_predictions = 0
    total_passengers = sum(titanic['Passengers'])
    
    for _, row in titanic.iterrows():
        group = row[group_by_attribute]
        predicted_outcome = majority_class[group]
        actual_survivors = row['Survivors']
        actual_deaths = row['Passengers'] - row['Survivors']
        
        if predicted_outcome == 'Survive':
            correct_predictions += actual_survivors
        else:
            correct_predictions += actual_deaths
    
    accuracy = correct_predictions / total_passengers
    return accuracy
majority_class_gender = get_majority_class(titanic, 'Gender')
training_accuracy = calculate_accuracy(titanic, 'Gender', majority_class_gender)


print(f'Training accuracy (only root node - Gender): {training_accuracy:.2%}') 
"""Training Accuracy = 77.6%"""

"""
Since we have the classes of passengers and survived, to get the training accuracy of this tree , we can get the majority-class for each combination and
calculate the training accuracy:
Upper = U
Lower = L
Male = M
Female = F
Child = C
Adult = L
+ = survived
- = dead
U+M+C = +
U+M+A = -
U+F+C = +
U+F+A = +
L+M+C = -
L+M+A = -
L+F+C = +
L+F+A = +
"""
correct_preditcions = 5 + 118 + 1 + 140 + 35 + 1211 + 27 + 176 
"""1713"""

training_accuracy_whole_tree = correct_preditcions / sum(titanic['Passengers'])
print(f'Training accuracy whole tree: {training_accuracy_whole_tree:.2%}') 