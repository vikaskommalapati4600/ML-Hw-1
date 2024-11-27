import numpy as np
import pandas as pd

# Gaussian (RBF) Kernel
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / gamma)

# Precompute the Gram matrix for the Gaussian kernel
def precompute_gram_matrix(X, gamma):
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K

# Kernel Perceptron Algorithm (Optimized with Gram Matrix)
def kernel_perceptron_optimized(X_train, y_train, X_test, y_test, gamma, max_epochs=100):
    n = len(X_train)
    alpha = np.zeros(n)
    train_errors = []
    
    # Precompute Gram matrix
    K_train = precompute_gram_matrix(X_train, gamma)

    # Training
    for epoch in range(max_epochs):
        errors = 0
        for i in range(n):
            prediction = np.sign(np.sum(alpha * y_train * K_train[:, i]))
            if prediction != y_train[i]:
                alpha[i] += 1
                errors += 1
        train_errors.append(errors / n)

        # Early stopping if no errors
        if errors == 0:
            break

    # Testing
    y_test_pred = []
    for x in X_test:
        pred = np.sign(
            sum(alpha[j] * y_train[j] * gaussian_kernel(X_train[j], x, gamma) for j in range(n))
        )
        y_test_pred.append(pred)
    y_test_pred = np.array(y_test_pred)
    test_error = np.mean(y_test_pred != y_test)

    return train_errors, test_error

# Load training and testing datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert labels to {-1, 1}
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Hyperparameter values
gamma_values = [0.1, 0.5, 1, 5, 100]

# Store results
results_optimized = []

for gamma in gamma_values:
    train_errors, test_error = kernel_perceptron_optimized(X_train, y_train, X_test, y_test, gamma)
    results_optimized.append({
        "gamma": gamma,
        "final_train_error": train_errors[-1],
        "test_error": test_error
    })

# Convert results to DataFrame for readability
results_optimized_df = pd.DataFrame(results_optimized)

# Display results
results_optimized_df.to_csv('kernal_perception.csv', index=True)

# Print results for console
print(results_optimized_df)
