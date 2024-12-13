import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def initialize_parameters(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)  
    parameters = {
        "W1": np.random.randn(input_size, hidden1_size) * 0.01,
        "b1": np.zeros((1, hidden1_size)),
        "W2": np.random.randn(hidden1_size, hidden2_size) * 0.01,
        "b2": np.zeros((1, hidden2_size)),
        "W3": np.random.randn(hidden2_size, output_size) * 0.01,
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
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache

def compute_loss(Y, A3):
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(A3) + (1 - Y) * np.log(1 - A3)) / m
    return loss

def back_propagation(X, Y, parameters, cache):
    m = X.shape[0]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    Z1, Z2, Z3 = cache["Z1"], cache["Z2"], cache["Z3"]
    
    dZ3 = A3 - Y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
    
    dZ2 = np.dot(dZ3, parameters["W3"].T) * sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dZ1 = np.dot(dZ2, parameters["W2"].T) * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return grads

def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    parameters["W3"] -= learning_rate * grads["dW3"]
    parameters["b3"] -= learning_rate * grads["db3"]
    return parameters

def train_neural_network(X, Y, input_size, hidden1_size, hidden2_size, output_size, epochs, learning_rate):
    parameters = initialize_parameters(input_size, hidden1_size, hidden2_size, output_size)
    for epoch in range(epochs):
        A3, cache = forward_propagation(X, parameters)
        loss = compute_loss(Y, A3)
        grads = back_propagation(X, Y, parameters, cache)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return parameters

def predict(X, parameters):
    A3, _ = forward_propagation(X, parameters)
    return (A3 > 0.5).astype(int)

train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

input_size = X_train.shape[1]
hidden1_size = 8  
hidden2_size = 8  
output_size = 1

trained_parameters = train_neural_network(X_train, y_train, input_size, hidden1_size, hidden2_size, output_size, epochs=1000, learning_rate=0.01)

predictions = predict(X_test, trained_parameters)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
