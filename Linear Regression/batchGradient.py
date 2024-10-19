import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file
train_data = pd.read_csv('/Users/vikaschowdary/Desktop/MLHW1/four/train.csv', header=None)

# Extract features and target from the dataset
data = train_data.values
X_train = data[:, :-1]  # Features (all columns except the last one)
y_train = data[:, -1]   # Target (last column)

# Adding a bias term to the feature matrix (intercept term)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

# Batch Gradient Descent function
def batch_gradient_descent(X, y, r=0.01, tolerance=1e-6, max_iterations=1000):
    m, n = X.shape
    w = np.zeros(n)  # Initialize weights
    cost_history = []
    weight_diff_history = []
    
    for iteration in range(max_iterations):
        # Calculate predictions
        predictions = X.dot(w)
        # Calculate error
        errors = predictions - y
        # Calculate cost (mean squared error)
        cost = (1/(2*m)) * np.dot(errors.T, errors)
        cost_history.append(cost)
        
        # Gradient calculation
        gradient = (1/m) * X.T.dot(errors)
        # Update weights
        w_new = w - r * gradient
        
        # Check for convergence
        weight_diff = np.linalg.norm(w_new - w)
        weight_diff_history.append(weight_diff)
        if weight_diff < tolerance:
            print(f"Convergence reached after {iteration} iterations.")
            break
        
        w = w_new
    
    return w, cost_history, weight_diff_history

# Applying the gradient descent with a learning rate of 0.5
r = 0.01
weights, cost_history, weight_diff_history = batch_gradient_descent(X_train, y_train, r=r)

# Plotting the cost function over iterations
plt.plot(cost_history)
plt.title("Cost Function Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# Displaying final weights and cost
print("Final weights:", weights)
print("Final cost:", cost_history[-1])
