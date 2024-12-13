import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def learning_rate_schedule(gamma0, t, d):
    return gamma0 / (1 + gamma0 * t / d)

def initialize_parameters_with_zeros(input_size, hidden_size, output_size):
    parameters = {
        "W1": np.zeros((input_size, hidden_size)),
        "b1": np.zeros((1, hidden_size)),
        "W2": np.zeros((hidden_size, hidden_size)),
        "b2": np.zeros((1, hidden_size)),
        "W3": np.zeros((hidden_size, output_size)),
        "b3": np.zeros((1, output_size))
    }
    return parameters

def forward_propagation(X, parameters):
    Z1 = np.dot(X, parameters["W1"]) + parameters["b1"]
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, parameters["W2"]) + parameters["b2"]
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, parameters["W3"]) + parameters["b3"]
    A3 = sigmoid(Z3)
    return A3, {"A1": A1, "A2": A2, "A3": A3}

def compute_error(y, y_pred):
    predictions = (y_pred > 0.5).astype(int)
    error = np.mean(predictions != y) * 100  
    return error

def back_propagation(X, y, parameters, cache):
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    m = X.shape[0]
    dZ3 = A3 - y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
    
    dZ2 = np.dot(dZ3, parameters["W3"].T) * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dZ1 = np.dot(dZ2, parameters["W2"].T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return grads

def update_parameters(parameters, grads, lr):
    for key in parameters.keys():
        parameters[key] -= lr * grads["d" + key]
    return parameters

def train_with_zeros(X_train, y_train, X_test, y_test, hidden_size, gamma0, d, epochs, batch_size):
    input_size, output_size = X_train.shape[1], 1
    parameters = initialize_parameters_with_zeros(input_size, hidden_size, output_size)
    t = 0  
    
    for epoch in range(epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[perm], y_train[perm]
        
        for i in range(0, X_train.shape[0], batch_size):
            t += 1
            lr = learning_rate_schedule(gamma0, t, d)
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            y_pred, cache = forward_propagation(X_batch, parameters)
            grads = back_propagation(X_batch, y_batch, parameters, cache)
            parameters = update_parameters(parameters, grads, lr)
    
    y_train_pred, _ = forward_propagation(X_train, parameters)
    y_test_pred, _ = forward_propagation(X_test, parameters)
    train_error = compute_error(y_train, y_train_pred)
    test_error = compute_error(y_test, y_test_pred)
    
    return train_error, test_error

train_data = pd.read_csv('/mnt/data/train.csv', header=None).values
test_data = pd.read_csv('/mnt/data/test.csv', header=None).values
X_train, y_train = train_data[:, :-1], train_data[:, -1].reshape(-1, 1)
X_test, y_test = test_data[:, :-1], test_data[:, -1].reshape(-1, 1)

hidden_widths = [5, 10, 25, 50, 100]
gamma0, d = 7e-5, 0.1
epochs, batch_size = 100, 32

print("\nTraining Results with Zero Weight Initialization:")
print(f"{'Hidden Width':<12} {'Training Error (%)':<20} {'Test Error (%)':<20}")
for width in hidden_widths:
    train_error, test_error = train_with_zeros(X_train, y_train, X_test, y_test, width, gamma0, d, epochs, batch_size)
    print(f"{width:<12} {train_error:.2f}%{' ' * 12}{test_error:.2f}%")
