import pandas as pd
import numpy as np
from pprint import pprint

# Read data
dataset = pd.read_csv('data/DecisionTree_data.csv')
data = pd.DataFrame(dataset, columns=['District', 'HouseType', 'Income', 'PreviousCustomer', 'Outcome'])
print(data)

# Set features
features = ['District', 'HouseType', 'Income', 'PreviousCustomer']
# Set target feature column
target_col = 'Outcome'


# Entropy calculation function
def entropy(target_col):
    element, count = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count))
         for i in range(len(element))])
    return entropy

# Information Gain Function
def InfoGain(data, attribute, target_col):
    # Calculate total entropy
    total_entropy = entropy(data[target_col])

    # Calculate weighted entropy
    element, count = np.unique(data[attribute], return_counts=True)
    child_entropy = np.sum(
        [(count[i] / len(data[attribute])) * entropy(
            data.where(data[attribute] == element[i]).dropna()[target_col])
         for i in range(len(element))])

    # Calculate information gain
    info_Gain = total_entropy - child_entropy
    # Return information gain
    return info_Gain


# Decision Tree function
def decisionTree(data, features, target_col):

    if len(data[target_col].value_counts()) == 1:
        return data[target_col].value_counts().index[0]

    # make tree
    else:
        item_values = [InfoGain(data, attribute, target_col) for attribute in features]
        # Return the information gain values for the attributes in the data
        best_attribute_index = np.argmax(item_values)
        best_attribute = features[best_attribute_index]

        # Create the tree structure.
        tree = {best_attribute: {}}
        features = [i for i in features if i != best_attribute]

        # Create sub tree
        for value in np.unique(data[best_attribute]):
            sub_data = data.where(data[best_attribute] == value).dropna()
            # Call 'decisionTree' Function for each  sub_data
            subtree = decisionTree(sub_data, features, target_col)
            # Add sub tree
            tree[best_attribute][value] = subtree

    return tree


print("===== Decision Tree =====")
tree = decisionTree(data, features, target_col)
pprint(tree)
