import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the training data
train_data = pd.read_csv('/Users/vikaschowdary/Desktop/MLHW1/four/train.csv', header=None)
data = train_data.values
X_train = data[:, :-1]  # Features
y_train = data[:, -1]   # Target

# Adding a bias term to the feature matrix
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

# Load the test data
test_data = pd.read_csv('/Users/vikaschowdary/Desktop/MLHW1/four/test.csv', header=None)
X_test = test_data.iloc[:, :-1].values  # Features for the test set
y_test = test_data.iloc[:, -1].values   # Target for the test set
X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # Adding bias term

# Stochastic Gradient Descent function
def stochastic_gradient_descent(X, y, r=0.01, tolerance=1e-6, max_iterations=5000):
    m, n = X.shape
    w = np.zeros(n)  # Initialize weights
    cost_history = []
    
    for iteration in range(max_iterations):
        for i in range(m):
            # Randomly select one sample
            rand_index = np.random.randint(m)
            xi = X[rand_index, :].reshape(1, -1)
            yi = y[rand_index].reshape(1)
            
            # Prediction and error for the selected sample
            prediction = xi.dot(w)
            error = prediction - yi
            
            # Gradient for the selected sample
            gradient = xi.T.dot(error)
            
            # Update weights
            w = w - r * gradient.flatten()
            
            # Calculate the cost (mean squared error) for the whole training set
            predictions = X.dot(w)
            errors = predictions - y
            cost = (1/(2*m)) * np.dot(errors.T, errors)
            cost_history.append(cost)
        
        # Check for convergence (optional)
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"Convergence reached after {iteration} iterations.")
            break
    
    return w, cost_history

# Applying the stochastic gradient descent with a learning rate of 0.01
r = 0.01
weights_sgd, cost_history_sgd = stochastic_gradient_descent(X_train, y_train, r=r, max_iterations=10)

# Plotting the cost function over iterations
plt.plot(cost_history_sgd)
plt.title("Cost Function Over Updates (SGD)")
plt.xlabel("Updates")
plt.ylabel("Cost")
plt.show()

# Making predictions on the test set using the final weights
y_test_pred_sgd = X_test.dot(weights_sgd)

# Calculating the cost (mean squared error) for the test data
m_test = len(y_test)
test_cost_sgd = (1 / (2 * m_test)) * np.sum((y_test_pred_sgd - y_test) ** 2)

print("Final weights (SGD):", weights_sgd)
print("Test cost (SGD):", test_cost_sgd)
