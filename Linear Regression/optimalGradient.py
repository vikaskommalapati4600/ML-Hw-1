import numpy as np
import pandas as pd

train_data = pd.read_csv('concrete/train.csv', header=None)
test_data = pd.read_csv('concrete/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

X_transpose = X_train.T
optimal_weights = np.linalg.inv(X_transpose.dot(X_train)).dot(X_transpose).dot(y_train)

y_test_pred_optimal = X_test.dot(optimal_weights)

m_test = len(y_test)
test_cost_optimal = (1 / (2 * m_test)) * np.sum((y_test_pred_optimal - y_test) ** 2)

print("Optimal Weights (Normal Equation):", optimal_weights)
print("Test Cost (Normal Equation):", test_cost_optimal)

weights_batch = np.array([-0.03471004, -0.0793754, -0.22597395, -0.2403659, 0.5083903, -0.03313133, 0.24581775, -0.0059534])
y_test_pred_batch = X_test.dot(weights_batch)
test_cost_batch = (1 / (2 * m_test)) * np.sum((y_test_pred_batch - y_test) ** 2)

print("\nBatch Gradient Descent Weights:", weights_batch)
print("Test Cost (Batch Gradient Descent):", test_cost_batch)

weights_sgd = np.array([-0.00808857, -0.09466168, -0.3147156, -0.19886576, 0.54574366, -0.01187641, 0.18451827, 0.0334366])
y_test_pred_sgd = X_test.dot(weights_sgd)
test_cost_sgd = (1 / (2 * m_test)) * np.sum((y_test_pred_sgd - y_test) ** 2)

print("\nStochastic Gradient Descent Weights:", weights_sgd)
print("Test Cost (Stochastic Gradient Descent):", test_cost_sgd)
