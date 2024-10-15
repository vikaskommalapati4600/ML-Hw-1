import pandas as pd
import numpy as np

def read_and_preprocess_replace_unknown(file_path, numerical_columns, columns):

    data = pd.read_csv(file_path, header=None, names=columns)


    for column in numerical_columns:
        median = data[column].median()
        data[column] = data[column].apply(lambda x: 1 if float(x) > median else 0)

    return data

def compute_entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def compute_majority_error(labels):
    _, counts = np.unique(labels, return_counts=True)
    return 1 - np.max(counts) / np.sum(counts)

def compute_gini_index(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities ** 2)

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

def build_decision_tree(data, original_data, features, target_attribute, heuristic, max_depth, current_depth=0):
    if len(np.unique(data[target_attribute])) == 1:
        return np.unique(data[target_attribute])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute])[np.argmax(np.unique(original_data[target_attribute], return_counts=True)[1])]
    elif len(features) == 0 or current_depth == max_depth:
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

def get_accuracy(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predictions = pd.Series([make_prediction(query, tree, 1) for query in queries])
    return (predictions == data.iloc[:, -1]).mean()

def experiment_with_trees(train_data, test_data, max_depths, heuristics):
    results = []
    features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

    for heuristic in heuristics:
        for max_depth in max_depths:
            tree = build_decision_tree(train_data, train_data, features, 'y', heuristic, max_depth)
            train_accuracy = get_accuracy(train_data, tree)
            test_accuracy = get_accuracy(test_data, tree)
            train_error = 1 - train_accuracy
            test_error = 1 - test_accuracy
            results.append((heuristic, max_depth, train_accuracy, test_accuracy, train_error, test_error))

    return results

columns = [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ]
numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

train_data = read_and_preprocess_replace_unknown('train.csv', numerical_columns, columns)
test_data = read_and_preprocess_replace_unknown('test.csv', numerical_columns, columns)

label_mapping = {'yes': 1, 'no': 0}
train_data['y'] = train_data['y'].map(label_mapping)
test_data['y'] = test_data['y'].map(label_mapping)

max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15, 16]
heuristics = ['entropy', 'majority_error', 'gini_index']

experiment_results = experiment_with_trees(train_data, test_data, max_depths, heuristics)

print(f"{'Heuristic':<20} {'Max Depth':<10} {'Train Accuracy':<15} {'Test Accuracy':<15} {'Train Error Rate':<18} {'Test Error Rate':<15}")
print("-" * 90)
for result in experiment_results:
    heuristic, max_depth, train_acc, test_acc, train_err, test_err = result
    print(f"{heuristic:<20} {max_depth:<10} {train_acc:<15.6f} {test_acc:<15.6f} {train_err:<18.6f} {test_err:<15.6f}")