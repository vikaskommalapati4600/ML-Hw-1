import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def build_decision_stump(data, original_data, features, target_attribute, heuristic):
    if len(np.unique(data[target_attribute])) == 1:
        return np.unique(data[target_attribute])[0]
    
    heuristic_values = [calculate_best_split(data, feature, target_attribute, heuristic) for feature in features]
    best_feature_index = np.argmax(heuristic_values)
    best_feature = features[best_feature_index]

    tree = {best_feature: {}}
    features = [f for f in features if f != best_feature]

    for value in np.unique(data[best_feature]):
        subset_data = data.where(data[best_feature] == value).dropna()
        tree[best_feature][value] = np.unique(subset_data[target_attribute])[np.argmax(np.unique(subset_data[target_attribute], return_counts=True)[1])]
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

# AdaBoost Algorithm
def adaboost(train_data, test_data, features, target_attribute, heuristic, T):
    n_samples = len(train_data)
    weights = np.ones(n_samples) / n_samples 
    stumps = []
    stump_weights = []
    train_errors = []
    test_errors = []
    stump_train_errors = []

    for t in range(T):
        stump = build_decision_stump(train_data, train_data, features, target_attribute, heuristic)

        predictions = np.array([make_prediction(row, stump, 1) for _, row in train_data.iterrows()])
        incorrect = (predictions != train_data[target_attribute])
        error = np.dot(weights, incorrect) / np.sum(weights)

        stump_weight = 0.5 * np.log((1 - error) / error)
        stump_weights.append(stump_weight)
        stumps.append(stump)

        weights *= np.exp(-stump_weight * ((predictions == train_data[target_attribute]) * 2 - 1))
        weights /= np.sum(weights)  

        train_errors.append(np.mean(predictions != train_data[target_attribute]))
        test_predictions = np.array([make_prediction(row, stump, 1) for _, row in test_data.iterrows()])
        test_errors.append(np.mean(test_predictions != test_data[target_attribute]))

        stump_train_errors.append(np.mean(predictions != train_data[target_attribute]))

    return stumps, stump_weights, train_errors, test_errors, stump_train_errors

def adaboost_predict(stumps, stump_weights, data):
    stump_predictions = np.array([[make_prediction(row, stump, 1) for stump in stumps] for _, row in data.iterrows()])
    weighted_predictions = np.dot(stump_weights, stump_predictions.T)
    return np.sign(weighted_predictions)

columns = [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ]
numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

train_data = read_and_preprocess_replace_unknown('train.csv', numerical_columns, columns)
test_data = read_and_preprocess_replace_unknown('test.csv', numerical_columns, columns)

label_mapping = {'yes': 1, 'no': 0}
train_data['y'] = train_data['y'].map(label_mapping)
test_data['y'] = test_data['y'].map(label_mapping)

T = 500  # Number of iterations increased to 500
features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
heuristic = 'entropy'

stumps, stump_weights, train_errors, test_errors, stump_train_errors = adaboost(train_data, test_data, features, 'y', heuristic, T)

iterations = list(range(1, T + 1))

plt.figure(figsize=(10, 6))
plt.plot(iterations, train_errors, label="Train Error", color='blue', marker='o')
plt.plot(iterations, test_errors, label="Test Error", color='orange', marker='x')
plt.xlabel("Number of Iterations (T)")
plt.ylabel("Error")
plt.title("Training and Test Errors vs Number of Iterations (T)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(iterations, stump_train_errors, label="Decision Stump Error", color='blue', marker='o')

plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Decision Stump Errors vs. Iteration")
plt.legend()
plt.grid(True)
plt.show()
