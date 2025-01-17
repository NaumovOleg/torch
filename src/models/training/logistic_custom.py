import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc

# Generate dataset
X, Y = make_classification(
    n_samples=100,
    n_features=2,  # Use 2 features for easy visualization
    n_classes=2,
    n_redundant=0,
    random_state=42,
    n_clusters_per_class=1,
    n_informative=2,
)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert Y to {0,1} instead of {0,1} for logistic regression
Y = Y.reshape(-1, 1)


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Logistic Regression Model (with Learning Rate)
def logistic_regression(X, Y, lr, epochs=1000):
    m, n = X.shape
    weights = np.zeros((n, 1))
    bias = 0
    losses = []

    for i in range(epochs):
        # Compute linear combination
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)

        # Compute loss (Binary Cross-Entropy)
        loss = (-1 / m) * np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))
        losses.append(loss)

        # Compute gradients
        dw = (1 / m) * np.dot(X.T, (y_pred - Y))
        db = (1 / m) * np.sum(y_pred - Y)

        # Update weights using gradient descent
        weights -= lr * dw
        bias -= lr * db

    return weights, bias, losses


# Different learning rates
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
loss_curves = []

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    _, _, losses = logistic_regression(X_scaled, Y, lr)
    loss_curves.append(losses)
    plt.plot(losses, label=f"LR = {lr}")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves for Different Learning Rates")
plt.legend()
plt.show()
