import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)
    X_train = train_df.iloc[:, :-1].values  # Features
    y_train = train_df.iloc[:, -1].values.reshape(-1, 1)  # Labels
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.reshape(-1, 1)
    return X_train, y_train, X_test, y_test

def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def backward_propagation(X, y, Z1, A1, Z2, A2, W1, W2):
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def train_nn(X, y, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        
        loss = np.mean((A2 - y) ** 2)
        
        dW1, db1, dW2, db2 = backward_propagation(X, y, Z1, A1, Z2, A2, W1, W2)
        
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
    return W1, b1, W2, b2, loss

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return (A2 > 0.5).astype(int)

def calculate_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

def print_results_table(hidden_sizes, train_errors, test_errors):
    print("\n| Hidden Width  | Training Error | Test Error |")
    print("|----|----------------|------------|")
    for i, h in enumerate(hidden_sizes):
        print(f"| {h:<3} | {train_errors[i]:<14.4f} | {test_errors[i]:<10.4f} |")

if __name__ == "__main__":
    train_file = 'train.csv'
    test_file = 'test.csv'
    X_train, y_train, X_test, y_test = load_data(train_file, test_file)
    
    hidden_sizes = [5, 10, 25, 50, 100]
    input_size = X_train.shape[1]
    output_size = 1
    learning_rate = 0.01
    epochs = 1000
    
    train_errors = []
    test_errors = []
    
    print("Training Neural Network with different hidden layer sizes...\n")
    for hidden_size in hidden_sizes:
        W1, b1, W2, b2, final_loss = train_nn(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs)
        
        train_predictions = predict(X_train, W1, b1, W2, b2)
        test_predictions = predict(X_test, W1, b1, W2, b2)
        train_error = calculate_error(y_train, train_predictions)
        test_error = calculate_error(y_test, test_predictions)
        
        train_errors.append(train_error)
        test_errors.append(test_error)

    print_results_table(hidden_sizes, train_errors, test_errors)
