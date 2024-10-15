import pandas as pd
import numpy as np

# Function to compute entropy (used for information gain)
def compute_entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Function to calculate majority error (what percentage of the dataset is not the majority class)
def compute_majority_error(labels):
    _, counts = np.unique(labels, return_counts=True)
    return 1 - np.max(counts) / np.sum(counts)

# Function to calculate the Gini Index (measures impurity)
def compute_gini_index(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities ** 2)

# Function to figure out the best heuristic based on user choice
def calculate_best_split(data, attribute, target_attribute, heuristic):
    total_samples = len(data)
    attribute_values, counts = np.unique(data[attribute], return_counts=True)

    if heuristic == 'entropy':
        total_entropy = compute_entropy(data[target_attribute])
        weighted_entropy = np.sum([(counts[i] / total_samples) * compute_entropy(data[data[attribute] == attribute_values[i]][target_attribute]) for i in range(len(attribute_values))])
        return total_entropy - weighted_entropy

    elif heuristic == 'majority_error':
        total_majority_error = compute_majority_error(data[target_attribute])
        weighted_majority_error = np.sum([(counts[i] / total_samples) * compute_majority_error(data[data[attribute] == attribute_values[i]][target_attribute]) for i in range(len(attribute_values))])
        return total_majority_error - weighted_majority_error

    elif heuristic == 'gini_index':
        total_gini = compute_gini_index(data[target_attribute])
        weighted_gini = np.sum([(counts[i] / total_samples) * compute_gini_index(data[data[attribute] == attribute_values[i]][target_attribute]) for i in range(len(attribute_values))])
        return total_gini - weighted_gini

# The ID3 algorithm to create a decision tree based on the selected heuristic and max depth
def build_decision_tree(data, original_data, features, target_attribute, heuristic, max_depth, current_depth=0):
    # Stop when all the labels are the same
    if len(np.unique(data[target_attribute])) == 1:
        return np.unique(data[target_attribute])[0]

    # If we have no more data, return the most common label from the original data
    elif len(data) == 0:
        return np.unique(original_data[target_attribute])[np.argmax(np.unique(original_data[target_attribute], return_counts=True)[1])]

    # If we've hit the max depth or there are no more features to split on
    elif len(features) == 0 or current_depth == max_depth:
        return np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]

    # Otherwise, continue building the tree
    else:
        most_common_label = np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]
        heuristic_values = [calculate_best_split(data, feature, target_attribute, heuristic) for feature in features]
        best_feature_index = np.argmax(heuristic_values)
        best_feature = features[best_feature_index]

        # Create a new tree node
        tree = {best_feature: {}}
        features = [f for f in features if f != best_feature]

        # Recursively build the subtrees for each value of the best feature
        for value in np.unique(data[best_feature]):
            subset_data = data.where(data[best_feature] == value).dropna()
            subtree = build_decision_tree(subset_data, original_data, features, target_attribute, heuristic, max_depth, current_depth + 1)
            tree[best_feature][value] = subtree
        return tree

# Function to make predictions using the decision tree
def make_prediction(query, tree, default=None):
    for key in list(query.keys()):
        if key in tree.keys():
            try:
                result = tree[key][query[key]]
            except:
                return default
            # If the result is still a dictionary, we need to go deeper in the tree
            if isinstance(result, dict):
                return make_prediction(query, result)
            else:
                return result

# Function to calculate the accuracy of the predictions on a dataset
def get_accuracy(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predictions = pd.Series([make_prediction(query, tree, 1) for query in queries])
    return (predictions == data.iloc[:, -1]).mean()

# Train and test decision trees for different depths and heuristics
def experiment_with_trees(train_data, test_data, max_depths, heuristics):
    results = []
    features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    for heuristic in heuristics:
        for max_depth in max_depths:
            tree = build_decision_tree(train_data, train_data, features, 'label', heuristic, max_depth)
            train_accuracy = get_accuracy(train_data, tree)
            test_accuracy = get_accuracy(test_data, tree)
            train_error = 1 - train_accuracy
            test_error = 1 - test_accuracy
            results.append((heuristic, max_depth, train_accuracy, test_accuracy, train_error, test_error))

    return results

# Load the training and testing datasets
train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

# Assign column names
train_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
test_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

# Convert label categories to numerical values
label_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
train_data['label'] = train_data['label'].map(label_mapping)
test_data['label'] = test_data['label'].map(label_mapping)

# Define the depths and heuristics to test
max_depths = [1, 2, 3, 4, 5, 6]
heuristics = ['entropy', 'majority_error', 'gini_index']

# Run the experiments
experiment_results = experiment_with_trees(train_data, test_data, max_depths, heuristics)

# Print the results in a formatted way using basic Python print functions
print(f"{'Heuristic':<20} {'Max Depth':<10} {'Train Accuracy':<15} {'Test Accuracy':<15} {'Train Error Rate':<18} {'Test Error Rate':<15}")
print("-" * 90)
for result in experiment_results:
    heuristic, max_depth, train_acc, test_acc, train_err, test_err = result
    print(f"{heuristic:<20} {max_depth:<10} {train_acc:<15.6f} {test_acc:<15.6f} {train_err:<18.6f} {test_err:<15.6f}")
