import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(w, X, y, v):
    m = X.shape[0]
    predictions = sigmoid(X @ w)
    gradient = -(X.T @ (y - predictions)) + (w / v)  
    return gradient / m

def logistic_regression_map(X_train, y_train, X_test, y_test, v, gamma_0, d, epochs=100):
    n_features = X_train.shape[1]
    w = np.zeros(n_features)  
    
    objective_curve = []
    
    for t in range(epochs):
        gamma_t = gamma_0 / (1 + (gamma_0 / d) * t)  
        for i in range(X_train.shape[0]):
            xi = X_train[i, :].reshape(1, -1)
            yi = y_train[i]
            grad = compute_gradient(w, xi, yi, v)
            w -= gamma_t * grad.flatten()
        predictions = sigmoid(X_train @ w)
        objective = -np.sum(yi * np.log(predictions) + (1 - yi) * np.log(1 - predictions)) + np.sum(w**2) / (2 * v)
        objective_curve.append(objective)
    
    train_preds = (sigmoid(X_train @ w) >= 0.5).astype(int)
    test_preds = (sigmoid(X_test @ w) >= 0.5).astype(int)
    train_error = np.mean(train_preds != y_train)
    test_error = np.mean(test_preds != y_test)
    
    return w, train_error, test_error, objective_curve

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
gamma_0 = 0.1
d = 1
epochs = 100

results = []
for v in variances:
    w, train_error, test_error, _ = logistic_regression_map(X_train, y_train, X_test, y_test, v, gamma_0, d, epochs)
    results.append({"Variance (v)": v, "Train Error": train_error, "Test Error": test_error})

results_df = pd.DataFrame(results)
results_df
