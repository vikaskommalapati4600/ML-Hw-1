import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preprocess the data
def read_and_preprocess_replace_unknown(file_path, numerical_columns, columns):
    data = pd.read_csv(file_path, header=None, names=columns)

    # Replace numerical columns by converting to binary based on the median
    for column in numerical_columns:
        median = data[column].median()
        data[column] = data[column].apply(lambda x: 1 if float(x) > median else 0)

    return data

# Calculate entropy
def compute_entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Calculate majority error
def compute_majority_error(labels):
    _, counts = np.unique(labels, return_counts=True)
    return 1 - np.max(counts) / np.sum(counts)

# Calculate Gini index
def compute_gini_index(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities ** 2)

# Calculate the best split for a given feature based on the heuristic
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

# Bootstrapping function to generate a random sample from the dataset
def bootstrap_sample(data):
    n_samples = len(data)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    sample = data.iloc[indices].reset_index(drop=True)
    return sample

# Bagged Trees Algorithm
def bagged_trees(train_data, test_data, features, target_attribute, n_trees):
    trees = []
    train_errors = []
    test_errors = []

    for i in range(1, n_trees + 1):
        # Bootstrap sampling from the training data
        sample = bootstrap_sample(train_data)

        # Train a fully-expanded decision tree on the bootstrap sample
        tree = build_decision_tree(sample, sample, features, target_attribute, heuristic='entropy', max_depth=None)

        # Store the trained tree
        trees.append(tree)

        # Make predictions using all trees (for both train and test sets)
        train_predictions = np.array([[make_prediction(row, tree, 1) for _, row in train_data.iterrows()] for tree in trees])
        test_predictions = np.array([[make_prediction(row, tree, 1) for _, row in test_data.iterrows()] for tree in trees])

        # Convert None to default value (1) for valid predictions
        train_predictions = np.where(train_predictions == None, 1, train_predictions)
        test_predictions = np.where(test_predictions == None, 1, test_predictions)

        # Majority vote for final predictions
        train_final_predictions = np.sign(np.mean(train_predictions, axis=0))
        test_final_predictions = np.sign(np.mean(test_predictions, axis=0))

        # Calculate and store errors
        train_errors.append(np.mean(train_final_predictions != train_data[target_attribute]))
        test_errors.append(np.mean(test_final_predictions != test_data[target_attribute]))

    return train_errors, test_errors

# Fully expanded decision tree (No max depth, no early stopping or pruning)
def build_decision_tree(data, original_data, features, target_attribute, heuristic, max_depth=None, current_depth=0):
    if len(np.unique(data[target_attribute])) == 1:
        return np.unique(data[target_attribute])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute])[np.argmax(np.unique(original_data[target_attribute], return_counts=True)[1])]
    elif len(features) == 0:
        return np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]
    else:
        heuristic_values = [calculate_best_split(data, feature, target_attribute, heuristic) for feature in features]
        best_feature_index = np.argmax(heuristic_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}
        features = [f for f in features if f != best_feature]

        for value in np.unique(data[best_feature]):
            subset_data = data.where(data[best_feature] == value).dropna()
            subtree = build_decision_tree(subset_data, original_data, features, target_attribute, heuristic, max_depth, current_depth + 1)
            tree[best_feature][value] = subtree
        return tree

# Prediction using the decision tree
def make_prediction(query, tree, default=None):
    for key in list(query.keys()):
        if key in tree.keys():
            try:
                result = tree[key][query[key]]
            except:
                return default
            if isinstance(result, dict):
                return make_prediction(query, result)
            else:
                return result
    return default

# Testing Bagged Trees
columns = [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ]
numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

# Read and preprocess the data
train_data = read_and_preprocess_replace_unknown('train.csv', numerical_columns, columns)
test_data = read_and_preprocess_replace_unknown('test.csv', numerical_columns, columns)

# Map labels
label_mapping = {'yes': 1, 'no': 0}
train_data['y'] = train_data['y'].map(label_mapping)
test_data['y'] = test_data['y'].map(label_mapping)

# Bagging parameters
n_trees = 500
features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
target_attribute = 'y'

# Train Bagged Trees
train_errors, test_errors = bagged_trees(train_data, test_data, features, target_attribute, n_trees)

# Plotting the train and test errors
iterations = list(range(1, n_trees + 1))

plt.figure(figsize=(8, 6))
plt.plot(iterations, train_errors, label="Train Error", color='blue', marker='o')
plt.plot(iterations, test_errors, label="Test Error", color='orange', marker='x')
plt.xlabel("Number of Trees")
plt.ylabel("Error")
plt.title("Training and Test Errors vs Number of Trees in Bagging")
plt.legend()
plt.grid(True)
plt.show()
