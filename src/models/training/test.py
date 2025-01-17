import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# from datasets import x_train as X, y_train as y

X, Y = make_classification(
    n_samples=120, n_features=2, n_classes=2, n_redundant=0, random_state=42
)

# X, y, *_ = make_classification(
#     n_samples=25,
#     n_features=2,
#     n_informative=2,
#     n_redundant=0,
#     random_state=11,
#     n_clusters_per_class=1,
#     class_sep=0.4,
# )

y = np.where(Y == 0, -1, 1)

X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

print(X)


def df(w, x, y):
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y


sigmoid = lambda z: 1 / (1 + np.exp(-z))
log_loss = lambda y, y_pred: -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

LR = 0.0001
N = 300000
weights = [0.0, 0.0, 0.0]
lm = 0.01


def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))


def gradient(w, x, y):
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y


quality = np.mean([loss(weights, x, y) for x, y in zip(X, y)])
Q = [quality]
for i in range(N):
    m, n = X.shape
    k = np.random.randint(0, len(X) - 1)
    current_x, current_y = X[k], y[k]
    weights = weights - LR * gradient(weights, current_x, current_y)

    quality = lm * loss(weights, current_x, current_y) + (1 - lm) * quality
    Q.append(quality)
    predicted = sigmoid(np.dot(current_x, weights))
    # if i % 1000 == 0:
    #     print("Quality: ", quality, "Predicted: ", predicted)

line_x = list(range(int(max(X[:, 0]))))
line_y = [-x * weights[0] / weights[1] - weights[2] / weights[2] for x in line_x]
x_0 = X[y == 1]
x_1 = X[y == -1]
plt.scatter(x_0[:, 0], x_0[:, 1], color="red")
plt.scatter(x_1[:, 0], x_1[:, 1], color="green")
plt.plot(line_x, line_y, color="blue")
plt.grid(True)
plt.show()


plt.plot(Q, color="b", label="Line Chart")
plt.grid(True)
plt.show()
