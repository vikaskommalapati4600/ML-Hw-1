import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

def average_perceptron_train(X, y, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0
    w_avg = np.zeros(X.shape[1])
    b_avg = 0
    counter = 0

    for _ in range(epochs):
        for i in range(len(X)):
            if y[i] * (np.dot(X[i], w) + b) <= 0:
                w += y[i] * X[i]
                b += y[i]
            w_avg += w
            b_avg += b
            counter += 1

    w_avg /= counter
    b_avg /= counter
    return w_avg, b_avg

w_avg, b_avg = average_perceptron_train(X_train, y_train, epochs=10)

def perceptron_predict(X, w, b):
    return np.where(np.dot(X, w) + b >= 0, 1, -1)

y_pred = perceptron_predict(X_test, w_avg, b_avg)

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print("Learned average weight vector:", w_avg)
print("Learned average bias term:", b_avg)
print("Average prediction error on test set:", error_rate)
