import numpy as np
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt

X, y, *_ = make_regression(n_samples=100, n_features=1, random_state=42, noise=10)
# X = np.array([1, 2, 3, 4, 5])
# y = np.array([40, 50, 60, 70, 80])

N = len(X)

sum_x = np.sum(X)
sum_y = np.sum(y)
sum_xy = np.sum(X * y)
sum_x_2 = np.sum(X**2)

print(np.sum(X * y))

W = ((N * sum_xy) - (sum_x * sum_y)) / (N * sum_x_2 - sum_x**2)
b = (sum_y - W * sum_x) / N

X_line = np.linspace(min(X), max(X), 100)
Y_line = W * X_line + b


future_X = 6
predicted_Y = W * future_X + b

# Plot the data
plt.scatter(X, y, color="blue", label="Actual Data", zorder=3)
plt.plot(X_line, Y_line, color="red", label="Regression Line", linewidth=2)
plt.scatter(
    future_X,
    predicted_Y,
    color="green",
    s=100,
)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
