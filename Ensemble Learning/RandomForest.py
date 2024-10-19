import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Generate Bootstrap Sample
def generate_bootstrap_sample(data):
    n_samples = len(data)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    sample = data.iloc[indices].reset_index(drop=True)
    return sample

# Calculate entropy
def compute_entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Build a decision tree with random feature selection for Random Forest
def build_decision_tree_random(data, original_data, features, target_attribute, feature_subset_size):
    if len(np.unique(data[target_attribute])) == 1:
        return np.unique(data[target_attribute])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute])[np.argmax(np.unique(original_data[target_attribute], return_counts=True)[1])]
    elif len(features) == 0:
        return np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]
    else:
        # Adjust feature subset size if it's larger than available features
        feature_subset_size = min(feature_subset_size, len(features))
        
        # Randomly select a subset of features
        random_features = np.random.choice(features, size=feature_subset_size, replace=False)
        heuristic_values = [compute_entropy(data[target_attribute]) for feature in random_features]
        best_feature_index = np.argmax(heuristic_values)
        best_feature = random_features[best_feature_index]

        tree = {best_feature: {}}
        features = [f for f in features if f != best_feature]

        for value in np.unique(data[best_feature]):
            subset_data = data.where(data[best_feature] == value).dropna()
            subtree = build_decision_tree_random(subset_data, original_data, features, target_attribute, feature_subset_size)
            tree[best_feature][value] = subtree
        return tree

# Prediction using the decision tree
def make_prediction(query, tree, default=1):
    for key in list(query.keys()):
        if key in tree.keys():
            try:
                result = tree[key][query[key]]
            except:
                return default
            if isinstance(result, dict):
                return make_prediction(query, result, default=default)
            else:
                return result
    return default

# Train a random forest with the random feature selection
def random_forest(train_data, features, target_attribute, num_trees, feature_subset_size):
    forest = []
    for _ in range(num_trees):
        # Bootstrap sampling from the training data
        bootstrap_sample = generate_bootstrap_sample(train_data)
        tree = build_decision_tree_random(bootstrap_sample, bootstrap_sample, features, target_attribute, feature_subset_size)
        forest.append(tree)
    return forest

# Prediction using the random forest (majority vote)
def predict_forest(forest, test_data):
    num_samples = len(test_data)
    predictions = np.zeros(num_samples)

    for tree in forest:
        tree_predictions = np.array([make_prediction(row, tree, default=1) for _, row in test_data.iterrows()])
        predictions += tree_predictions

    return np.sign(predictions)  # Majority vote

# Preprocess the data
def read_and_preprocess_replace_unknown(file_path, numerical_columns, columns):
    data = pd.read_csv(file_path, header=None, names=columns)

    for column in numerical_columns:
        median = data[column].median()
        data[column] = data[column].apply(lambda x: 1 if float(x) > median else 0)

    return data

# Evaluate the random forest
def evaluate_random_forest(train_data, test_data, features, target_attribute, num_trees_list, feature_subset_sizes):
    for feature_subset_size in feature_subset_sizes:
        train_errors = []
        test_errors = []

        for num_trees in num_trees_list:
            forest = random_forest(train_data, features, target_attribute, num_trees, feature_subset_size)

            # Training error
            train_predictions = predict_forest(forest, train_data)
            train_accuracy = np.mean(train_predictions == train_data[target_attribute].values)
            train_errors.append(1 - train_accuracy)

            # Test error
            test_predictions = predict_forest(forest, test_data)
            test_accuracy = np.mean(test_predictions == test_data[target_attribute].values)
            test_errors.append(1 - test_accuracy)

        # Plot the results
        plt.plot(num_trees_list, train_errors, label=f"Train Error (Feature Size={feature_subset_size})")
        plt.plot(num_trees_list, test_errors, label=f"Test Error (Feature Size={feature_subset_size})")

    plt.xlabel("Number of Trees")
    plt.ylabel("Error")
    plt.title("Training and Test Errors vs Number of Trees (Random Forest)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Testing the Random Forest implementation
columns = [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ]
numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

train_data = read_and_preprocess_replace_unknown('train.csv', numerical_columns, columns)
test_data = read_and_preprocess_replace_unknown('test.csv', numerical_columns, columns)

label_mapping = {'yes': 1, 'no': 0}
train_data['y'] = train_data['y'].map(label_mapping)
test_data['y'] = test_data['y'].map(label_mapping)

features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
target_attribute = 'y'

# Full evaluation with original number of trees and feature subset sizes
num_trees_list = [1, 10, 50, 100, 200, 300, 400, 500]  # Full range of trees
feature_subset_sizes = [2, 4, 6]  # Vary feature subset size as required

evaluate_random_forest(train_data, test_data, features, target_attribute, num_trees_list, feature_subset_sizes)
