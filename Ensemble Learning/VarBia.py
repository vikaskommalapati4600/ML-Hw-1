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

# Build a fully expanded decision tree
def build_decision_tree(data, original_data, features, target_attribute):
    if len(np.unique(data[target_attribute])) == 1:
        return np.unique(data[target_attribute])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute])[np.argmax(np.unique(original_data[target_attribute], return_counts=True)[1])]
    elif len(features) == 0:
        return np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]
    else:
        heuristic_values = [compute_entropy(data[target_attribute]) for feature in features]
        best_feature_index = np.argmax(heuristic_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}
        features = [f for f in features if f != best_feature]

        for value in np.unique(data[best_feature]):
            subset_data = data.where(data[best_feature] == value).dropna()
            subtree = build_decision_tree(subset_data, original_data, features, target_attribute)
            tree[best_feature][value] = subtree
        return tree

# Prediction using the decision tree
def make_prediction(query, tree, default=1):
    for key in list(query.keys()):
        if key in tree.keys():
            try:
                result = tree[key][query[key]]
            except:
                return default  # Return default value if prediction cannot be made
            if isinstance(result, dict):
                return make_prediction(query, result, default=default)
            else:
                return result
    return default  # Return default if no prediction is possible

# Bootstrapping function to generate random samples
def bootstrap_sample(data):
    n_samples = len(data)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    sample = data.iloc[indices].reset_index(drop=True)
    return sample

# Bagged Trees Algorithm
def bagged_trees(train_data, features, target_attribute, n_trees):
    trees = []

    for _ in range(n_trees):
        # Bootstrap sampling from the training data
        sample = bootstrap_sample(train_data)
        tree = build_decision_tree(sample, sample, features, target_attribute)
        trees.append(tree)

    return trees

# Bias-Variance Decomposition
def bias_variance_decomposition(train_data, test_data, features, target_attribute, n_trees=10, n_repeats=100):
    single_tree_predictions = []
    bagged_predictions = []

    # Repeat the procedure n_repeats times
    for _ in range(n_repeats):
        # Step 1: Sample 1,000 examples without replacement from the training data
        sample = train_data.sample(n=1000, replace=False)

        # Step 2: Train 500 bagged trees on the sampled data
        trees = bagged_trees(sample, features, target_attribute, n_trees)

        # Store predictions for single trees (take the first tree from each bag)
        single_tree_preds = [make_prediction(row, trees[0], default=1) for _, row in test_data.iterrows()]
        single_tree_predictions.append(single_tree_preds)

        # Store predictions for bagged trees (majority vote from all trees)
        bagged_tree_preds = []
        for _, row in test_data.iterrows():
            predictions = np.array([make_prediction(row, tree, default=1) for tree in trees])
            majority_vote = np.sign(np.sum(predictions))  # Majority vote
            bagged_tree_preds.append(majority_vote)
        bagged_predictions.append(bagged_tree_preds)

    # Convert to numpy arrays for easier manipulation
    single_tree_predictions = np.array(single_tree_predictions)
    bagged_predictions = np.array(bagged_predictions)

    # Ground truth labels for test data
    ground_truth = test_data[target_attribute].values

    # Bias and variance for single trees
    single_tree_avg_pred = np.mean(single_tree_predictions, axis=0)
    single_tree_bias = np.mean((single_tree_avg_pred - ground_truth) ** 2)
    single_tree_variance = np.mean(np.var(single_tree_predictions, axis=0))

    # Bias and variance for bagged trees
    bagged_tree_avg_pred = np.mean(bagged_predictions, axis=0)
    bagged_tree_bias = np.mean((bagged_tree_avg_pred - ground_truth) ** 2)
    bagged_tree_variance = np.mean(np.var(bagged_predictions, axis=0))

    return {
        'single_tree_bias': single_tree_bias,
        'single_tree_variance': single_tree_variance,
        'bagged_tree_bias': bagged_tree_bias,
        'bagged_tree_variance': bagged_tree_variance
    }

# Testing Bias-Variance Decomposition
columns = [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ]
numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

# Read and preprocess the data
train_data = read_and_preprocess_replace_unknown('train.csv', numerical_columns, columns)
test_data = read_and_preprocess_replace_unknown('test.csv', numerical_columns, columns)

# Map labels
label_mapping = {'yes': 1, 'no': 0}
train_data['y'] = train_data['y'].map(label_mapping)
test_data['y'] = test_data['y'].map(label_mapping)

# Bias-Variance Analysis
features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
target_attribute = 'y'

results = bias_variance_decomposition(train_data, test_data, features, target_attribute, n_trees=500, n_repeats=100)

# Output the results
print("Single Tree Bias:", results['single_tree_bias'])
print("Single Tree Variance:", results['single_tree_variance'])
print("Bagged Tree Bias:", results['bagged_tree_bias'])
print("Bagged Tree Variance:", results['bagged_tree_variance'])
