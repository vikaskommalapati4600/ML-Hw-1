import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)


X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

def voted_perceptron_train(X, y, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0
    weight_vectors = []
    count = 1

    for _ in range(epochs):
        for i in range(len(X)):
            if y[i] * (np.dot(X[i], w) + b) <= 0:
                weight_vectors.append((w.copy(), b, count))
                w += y[i] * X[i]
                b += y[i]
                count = 1
            else:
                count += 1

    weight_vectors.append((w, b, count))
    return weight_vectors

weight_vectors = voted_perceptron_train(X_train, y_train, epochs=10)

def voted_perceptron_predict(X, weight_vectors):
    predictions = []
    for x in X:
        vote = sum(count * np.sign(np.dot(w, x) + b) for w, b, count in weight_vectors)
        predictions.append(np.sign(vote))
    return np.array(predictions)

y_pred = voted_perceptron_predict(X_test, weight_vectors)

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print("List of distinct weight vectors and their counts:")
for i, (w, b, count) in enumerate(weight_vectors):
    print(f"Weight vector {i + 1}: {w}, Bias: {b}, Count: {count}")

print("Average prediction error on test set:", error_rate)
