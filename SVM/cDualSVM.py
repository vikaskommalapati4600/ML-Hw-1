import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load datasets
train_data = pd.read_csv('train.csv')
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# Convert labels to {-1, 1}
y_train = np.where(y_train == 0, -1, 1)

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

# Reduce dataset size for faster computation (random subsampling)
np.random.seed(42)
sample_size = 300  # Reduced dataset size
indices = np.random.choice(range(len(X_train)), sample_size, replace=False)

X_train_small = X_train[indices]
y_train_small = y_train[indices]

# Hyperparameter values
gamma_values = [0.01, 0.1, 0.5, 1, 5, 100]  # Updated list to include 0.01
C_value = 500 / 873

# Store support vector indices and performance metrics for the updated gamma values
support_vectors_dict = {}

for gamma in gamma_values:
    # Compute Gram matrix for the smaller dataset
    K_train_small = compute_gram_matrix(X_train_small, gamma)
    
    # Optimize dual problem for given gamma and C
    bounds = [(0, C_value) for _ in range(len(y_train_small))]
    constraints = [{'type': 'eq', 'fun': eq_constraint, 'args': (y_train_small,)}]
    initial_alpha = np.zeros(len(y_train_small))
    
    result = minimize(
        dual_objective,
        x0=initial_alpha,
        args=(K_train_small, y_train_small),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 100, 'ftol': 1e-3}  # Relaxed tolerance for faster convergence
    )
    
    alpha = result.x
    
    # Get support vector indices
    support_vectors = np.where(alpha > 1e-5)[0]
    support_vectors_dict[gamma] = set(support_vectors)

# Compare consecutive gamma values
overlap_results = []
for i in range(len(gamma_values) - 1):
    gamma1, gamma2 = gamma_values[i], gamma_values[i + 1]
    overlap = support_vectors_dict[gamma1] & support_vectors_dict[gamma2]
    overlap_results.append({
        "gamma1": gamma1,
        "gamma2": gamma2,
        "overlap_count": len(overlap),
        "support_vectors_gamma1": len(support_vectors_dict[gamma1]),
        "support_vectors_gamma2": len(support_vectors_dict[gamma2])
    })

# Convert results to a DataFrame for readability
overlap_df = pd.DataFrame(overlap_results)

# Display results
overlap_df.to_csv('updated_svm_overlap.csv', index=True)

# Print the results to console
print(overlap_df)
