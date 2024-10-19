import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Function to generate bootstrap sample
def generate_bootstrap_sample(data):
    n_samples = len(data)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    sample = data.iloc[indices].reset_index(drop=True)
    return sample

# Function to compute entropy
def compute_entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Function to build a random decision tree
def build_decision_tree_random(data, original_data, features, target_attribute, feature_subset_size):
    if len(np.unique(data[target_attribute])) == 1:
        return np.unique(data[target_attribute])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute])[np.argmax(np.unique(original_data[target_attribute], return_counts=True)[1])]
    elif len(features) == 0:
        return np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]
    else:
        feature_subset_size = min(feature_subset_size, len(features))
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

# Train a random forest with random feature selection
def random_forest(train_data, features, target_attribute, num_trees, feature_subset_size):
    forest = []
    for _ in range(num_trees):
        bootstrap_sample = generate_bootstrap_sample(train_data)
        tree = build_decision_tree_random(bootstrap_sample, bootstrap_sample, features, target_attribute, feature_subset_size)
        forest.append(tree)
    return forest

# Predict using the random forest
def predict_forest(forest, test_data):
    num_samples = len(test_data)
    predictions = np.zeros(num_samples)
    
    for tree in forest:
        tree_predictions = np.array([make_prediction(row, tree, default=1) for _, row in test_data.iterrows()])
        predictions += tree_predictions
    
    return np.sign(predictions)  # Majority vote

# Bias-variance decomposition for a single random tree and the entire forest
def bias_variance_decomposition(train_data, test_data, features, target_attribute, num_trees=10, n_repeats=10, feature_subset_size=3):
    single_tree_predictions = []
    forest_predictions = []

    for _ in range(n_repeats):
        # Step 1: Generate a random forest
        forest = random_forest(train_data, features, target_attribute, num_trees, feature_subset_size)

        # Step 2: Store predictions for single trees and the forest
        single_tree_preds = [make_prediction(row, forest[0], default=1) for _, row in test_data.iterrows()]
        single_tree_predictions.append(single_tree_preds)

        # Majority vote for the forest predictions
        forest_preds = predict_forest(forest, test_data)
        forest_predictions.append(forest_preds)

    single_tree_predictions = np.array(single_tree_predictions)
    forest_predictions = np.array(forest_predictions)

    # Ground truth
    ground_truth = test_data[target_attribute].values

    # Compute bias and variance for single tree
    single_tree_avg_pred = np.mean(single_tree_predictions, axis=0)
    single_tree_bias = np.mean((single_tree_avg_pred - ground_truth) ** 2)
    single_tree_variance = np.mean(np.var(single_tree_predictions, axis=0))

    # Compute bias and variance for the whole forest
    forest_avg_pred = np.mean(forest_predictions, axis=0)
    forest_bias = np.mean((forest_avg_pred - ground_truth) ** 2)
    forest_variance = np.mean(np.var(forest_predictions, axis=0))

    return {
        'single_tree_bias': single_tree_bias,
        'single_tree_variance': single_tree_variance,
        'forest_bias': forest_bias,
        'forest_variance': forest_variance
    }

# Preprocess the data
def read_and_preprocess_replace_unknown(file_path, numerical_columns, columns):
    data = pd.read_csv(file_path, header=None, names=columns)

    for column in numerical_columns:
        median = data[column].median()
        data[column] = data[column].apply(lambda x: 1 if float(x) > median else 0)

    return data

# Testing the Random Forest Bias-Variance Decomposition
columns = [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ]
numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

train_data = read_and_preprocess_replace_unknown('train.csv', numerical_columns, columns)
test_data = read_and_preprocess_replace_unknown('test.csv', numerical_columns, columns)

# Map labels
label_mapping = {'yes': 1, 'no': 0}
train_data['y'] = train_data['y'].map(label_mapping)
test_data['y'] = test_data['y'].map(label_mapping)

features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
target_attribute = 'y'

# Perform Bias-Variance Decomposition
results = bias_variance_decomposition(train_data, test_data, features, target_attribute, num_trees=10, n_repeats=10, feature_subset_size=3)

# Output results
print("Single Tree Bias:", results['single_tree_bias'])
print("Single Tree Variance:", results['single_tree_variance'])
print("Random Forest Bias:", results['forest_bias'])
print("Random Forest Variance:", results['forest_variance'])
