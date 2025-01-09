import numpy as np
import matplotlib.pyplot as plt

from datasets import x_train, y_train

x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)))


# sigmoid loss function
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))


def gradient(w, x, y):
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(w, X, threshold=0.5):
    z = np.dot(X, w)
    # Apply the sigmoid function to get probabilities
    probabilities = sigmoid(z)
    # Convert probabilities to binary predictions
    predictions = (probabilities >= threshold).astype(int)
    return predictions


n_train = len(x_train)
w = [0.0, 0.0, 0.0]  # initial value of  weights
nt = 0.0005  # шаг сходимости SGD

N = 5000
# lm = 2 / (N + 1)  # 0.01
lm = 0.01  # скорость "забывания" для Q


Q = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])  # показатель качества
Q_plot = [Q]
Q_mean_gradient = np.mean([gradient(w, x, y) for x, y in zip(x_train, y_train)], axis=0)
Q_gradients = np.array([gradient(w, x, y) for x, y in zip(x_train, y_train)])

print("Q_mean_gradient", Q_mean_gradient)
print("Q_gradients", Q_gradients)
print("--------------------", len(Q_gradients))


def recalculate_gradients(index, weights, x, y):
    new_gradient = gradient(weights, x, y)
    new_mean_gradient = (
        Q_mean_gradient
        - (Q_gradients[index] / len(Q_gradients))
        + (new_gradient / len(Q_gradients))
    )
    Q_gradients[index] = new_gradient
    return new_mean_gradient


for i in range(N):
    k = np.random.randint(0, n_train - 1)
    ek = loss(w, x_train[k], y_train[k])
    Q_mean_gradient = recalculate_gradients(k, w, x_train[k], y_train[k])
    w = w - nt * Q_mean_gradient
    Q = lm * ek + (1 - lm) * Q
    Q_plot.append(Q)
    if Q <= 0.03:
        break

print("QQQQQ", Q)


line_x = list(range(max(x_train[:, 0])))
line_y = [-x * w[0] / w[1] - w[2] / w[2] for x in line_x]
line_z = [-x * w[0] / w[1] - w[2] / w[2] for x in line_x]
x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]
plt.scatter(x_0[:, 0], x_0[:, 1], color="red")
plt.scatter(x_1[:, 0], x_1[:, 1], color="green")
plt.plot(line_x, line_y, color="blue")
plt.grid(True)
plt.show()

plt.plot(Q_plot, color="b", label="Line Chart")
plt.grid(True)
plt.show()
