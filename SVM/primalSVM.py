import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, y_train)
y_test = np.where(y_test == 0, -1, y_test)

def svm_sgd(X, y, C, learning_rate_schedule, T=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    objective_values = []

    for epoch in range(T):
        indices = np.random.permutation(n_samples)  
        for t, i in enumerate(indices):
            lr = learning_rate_schedule(t + epoch * n_samples)
            margin = y[i] * (np.dot(w, X[i]) + b)
            if margin < 1:
                w = (1 - lr) * w + lr * C * y[i] * X[i]
                b += lr * C * y[i]
            else:
                w = (1 - lr) * w
        objective = 0.5 * np.dot(w, w) + C * np.sum(np.maximum(0, 1 - y * (np.dot(X, w) + b)))
        objective_values.append(objective)
    return w, b, objective_values

def schedule_a(t, gamma_0, a):
    return gamma_0 / (1 + gamma_0 * t / a)

def schedule_b(t, gamma_0):
    return gamma_0 / (1 + t)

def evaluate(X, y, w, b):
    predictions = np.sign(np.dot(X, w) + b)
    accuracy = np.mean(predictions == y)
    return 1 - accuracy  

C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_0 = 0.1
a = 10

results = {}

for C in C_values:
    w_a, b_a, objective_a = svm_sgd(
        X_train, y_train, C, lambda t: schedule_a(t, gamma_0, a)
    )
    train_error_a = evaluate(X_train, y_train, w_a, b_a)
    test_error_a = evaluate(X_test, y_test, w_a, b_a)

    w_b, b_b, objective_b = svm_sgd(
        X_train, y_train, C, lambda t: schedule_b(t, gamma_0)
    )
    train_error_b = evaluate(X_train, y_train, w_b, b_b)
    test_error_b = evaluate(X_test, y_test, w_b, b_b)

    results[C] = {
        "schedule_a": {"train_error": train_error_a, "test_error": test_error_a, "objective": objective_a},
        "schedule_b": {"train_error": train_error_b, "test_error": test_error_b, "objective": objective_b},
        "model_diff": np.linalg.norm(w_a - w_b),
        "error_diff": abs(train_error_a - train_error_b),
    }

for C, res in results.items():
    print(f"Results for C = {C:.2f}:")
    print("  Schedule A:")
    print(f"    - Training error: {res['schedule_a']['train_error'] * 100:.2f}%")
    print(f"    - Test error: {res['schedule_a']['test_error'] * 100:.2f}%")
    print(f"    - Model parameters: The model with schedule A has weights adjusted using a decay schedule "
          f"to balance regularization and fitting.")
    print(f"    - Interpretation: The training and test errors indicate the model's ability to fit the data "
          f"while maintaining generalization. Lower error suggests better generalization.\n")

    print("  Schedule B:")
    print(f"    - Training error: {res['schedule_b']['train_error'] * 100:.2f}%")
    print(f"    - Test error: {res['schedule_b']['test_error'] * 100:.2f}%")
    print(f"    - Model parameters: The model with schedule B uses a different decay schedule, resulting in "
          f"different convergence behavior.")
    print(f"    - Interpretation: The comparison between schedules highlights the impact of learning rate "
          f"decay on generalization performance. Lower errors generally reflect better performance.\n")

    print(f"Comparison between Schedules for C = {C:.2f}:")
    print(f"  - Difference in model weights: {res['model_diff']:.4f}")
    print(f"  - Difference in training error: {res['error_diff'] * 100:.2f}%\n")

for C in C_values:
    plt.figure()
    plt.plot(results[C]["schedule_a"]["objective"], label=f"Schedule A (C={C})")
    plt.xlabel("Epochs")
    plt.ylabel("Objective Function Value")
    plt.legend()
    plt.title(f"Convergence of Objective Function (Schedule A, C={C})")
    plt.savefig(f"schedule_a_convergence_C_{C:.3f}.png")  
    plt.show()

for C in C_values:
    plt.figure()
    plt.plot(results[C]["schedule_b"]["objective"], label=f"Schedule B (C={C})")
    plt.xlabel("Epochs")
    plt.ylabel("Objective Function Value")
    plt.legend()
    plt.title(f"Convergence of Objective Function (Schedule B, C={C})")
    plt.savefig(f"schedule_b_convergence_C_{C:.3f}.png")  
    plt.show()
