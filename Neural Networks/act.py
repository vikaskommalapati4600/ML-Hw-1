import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate  

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, depth, activation_fn):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(depth - 1):
            layers.append(activation_fn)
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation_fn)
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Sigmoid())  
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def learning_rate_schedule(gamma_0, d, t):
    return gamma_0 / (1 + (gamma_0 / d) * t)

def train_model(model, train_loader, criterion, gamma_0, d, epochs):
    optimizer = optim.Adam(model.parameters(), lr=gamma_0)
    for epoch in range(epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            t = epoch * len(train_loader) + batch_idx
            lr = learning_rate_schedule(gamma_0, d, t)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = criterion(y_pred, y)
    return loss.item()

train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]
gamma_0 = 1e-3
d = 0.01
epochs = 100

results = []

for activation_name, activation_fn, init_fn in [('tanh', nn.Tanh(), xavier_init), ('relu', nn.ReLU(), he_init)]:
    for width in widths:
        for depth in depths:
            print(f"Training with activation: {activation_name}, hidden size: {width}, depth: {depth}")

            model = NeuralNetwork(X_train_tensor.shape[1], 1, width, depth, activation_fn)
            model.apply(init_fn)

            criterion = nn.BCELoss()
            train_model(model, train_loader, criterion, gamma_0, d, epochs)

            train_error = evaluate_model(model, X_train_tensor, y_train_tensor, criterion)
            val_error = evaluate_model(model, X_val_tensor, y_val_tensor, criterion)

            results.append([activation_name, width, depth, f"{train_error:.4f}", f"{val_error:.4f}"])

headers = ["Activation", "Width", "Depth", "Train Error", "Validation Error"]
print("\nFinal Results:")
print(tabulate(results, headers=headers, tablefmt="grid"))
