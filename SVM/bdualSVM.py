import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert labels to {-1, 1}
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Gaussian (RBF) Kernel
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / gamma)

# Compute the Gram matrix for Gaussian kernel
def compute_gram_matrix(X, gamma):
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K

# Dual objective function
def dual_objective(alpha, K, y):
    return 0.5 * np.sum(
        np.outer(alpha, alpha) * np.outer(y, y) * K
    ) - np.sum(alpha)

# Equality constraint: sum(alpha * y) = 0
def eq_constraint(alpha, y):
    return np.dot(alpha, y)

# Calculate weights and bias
def calculate_bias(alpha, K, y, C):
    support_vectors = (alpha > 1e-5)
    b = np.mean(
        y[support_vectors] - np.dot(K[support_vectors][:, support_vectors], alpha[support_vectors] * y[support_vectors])
    )
    return b, support_vectors

# Prediction function
def predict(X, X_train, y_train, alpha, b, gamma):
    predictions = []
    for x in X:
        prediction = np.sum(
            alpha * y_train * np.array([gaussian_kernel(x, x_train, gamma) for x_train in X_train])
        ) + b
        predictions.append(np.sign(prediction))
    return np.array(predictions)

# Hyperparameter values
gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100/873, 500/873, 700/873]

results = []

for gamma in gamma_values:
    K_train = compute_gram_matrix(X_train, gamma)
    for C in C_values:
        bounds = [(0, C) for _ in range(len(y_train))]
        constraints = [{'type': 'eq', 'fun': eq_constraint, 'args': (y_train,)}]
        initial_alpha = np.zeros(len(y_train))

        # Solve the dual problem
        result = minimize(
            dual_objective,
            x0=initial_alpha,
            args=(K_train, y_train),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        alpha = result.x
        b, support_vectors = calculate_bias(alpha, K_train, y_train, C)

        # Training and test predictions
        y_train_pred = predict(X_train, X_train, y_train, alpha, b, gamma)
        y_test_pred = predict(X_test, X_train, y_train, alpha, b, gamma)

        # Calculate errors
        train_error = np.mean(y_train_pred != y_train)
        test_error = np.mean(y_test_pred != y_test)

        results.append({
            "gamma": gamma,
            "C": C,
            "train_error": train_error,
            "test_error": test_error,
            "support_vectors": np.sum(support_vectors)
        })

# Convert results to a DataFrame for better analysis
results_df = pd.DataFrame(results)
results_df.to_csv('non_linear_svm_results.csv', index=True)
print(results_df)

