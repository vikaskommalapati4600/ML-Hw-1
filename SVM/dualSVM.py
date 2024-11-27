import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load the training and testing data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Extract features (X) and labels (y) from training data
X_train = train_data.iloc[:, :-1].values  # All columns except the last one
y_train = train_data.iloc[:, -1].values   # The last column

# Extract features and labels from the test data
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Ensure labels are in the range {-1, 1}
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Define the kernel function (linear kernel for simplicity)
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Create the Gram matrix
N = len(y_train)
K = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i, j] = linear_kernel(X_train[i], X_train[j])

# Objective function for dual SVM
def objective(alpha):
    return 0.5 * np.sum(
        np.outer(alpha, alpha) * np.outer(y_train, y_train) * K
    ) - np.sum(alpha)

# Equality constraint: sum(alpha * y) = 0
def eq_constraint(alpha):
    return np.dot(alpha, y_train)

# SVM parameters
C_values = [100/873, 500/873, 700/873]  # Regularization parameters
results = {}

# Function to calculate weights, bias, and support vectors
def calculate_weights_bias(alphas, X, y):
    w = np.sum(alphas[:, None] * y[:, None] * X, axis=0)
    support_vectors = alphas > 1e-5
    b = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))
    return w, b, support_vectors

# Function to calculate prediction error
def calculate_error(X, y, w, b):
    predictions = np.sign(np.dot(X, w) + b)
    error = np.mean(predictions != y)
    return error

# Solve for each C
for C in C_values:
    bounds = [(0, C) for _ in range(N)]
    constraints = [{'type': 'eq', 'fun': eq_constraint}]
    
    # Optimize
    result = minimize(
        objective, 
        x0=np.zeros(N), 
        bounds=bounds, 
        constraints=constraints, 
        method='SLSQP'
    )
    alphas = result.x
    
    # Recover weights, bias, and support vectors
    w, b, support_vectors = calculate_weights_bias(alphas, X_train, y_train)
    
    # Calculate training and testing errors
    train_error = calculate_error(X_train, y_train, w, b)
    test_error = calculate_error(X_test, y_test, w, b)
    
    results[C] = {
        "w": w,
        "b": b,
        "support_vectors": np.sum(support_vectors),
        "train_error": train_error,
        "test_error": test_error
    }

# Convert results to a DataFrame for better presentation
results_df = pd.DataFrame(results).T
results_df.to_csv('svm_results_with_errors.csv', index=True)
print(results_df)
