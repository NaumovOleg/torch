import numpy as np
import matplotlib.pyplot as plt

from datasets import x_train, y_train

x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)))


# sigmoid loss function
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))


def df(w, x, y):
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y


n_train = len(x_train)
w = [0.0, 0.0, 0.0]  # initial value of  weights
nt = 0.0005  # шаг сходимости SGD
# скорость "забывания" для Q
N = 500
lm = 2 / (N + 1)  # 0.01

T_speed = 20

Q = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])  # показатель качества
Q_plot = [Q]


for i in range(N):
    k = np.random.randint(0, n_train - 1)  # random index

    learning_coefficient = np.exp(-(i / N))
    # print(learning_coefficient)
    # lm = lm * learning_coefficient
    indexes = np.random.randint(low=0, high=n_train, size=7, dtype=int)
    ek = np.mean([loss(w, x_train[o], y_train[o]) for o in indexes])
    loss_value = loss(w, x_train[k], y_train[k])

    w = w - nt * df(w, x_train[k], y_train[k])  # Псевдо градиентный алгоритм
    Q = lm * loss_value + (1 - lm) * Q  # скользящее экспоненциальное сглаживание
    Q_plot.append(Q)

    # ek = loss(w, x_train[k], y_train[k])  # вычисление потерь для выбранного вектора
    # w = w - nt * df(w, x_train[k], y_train[k])  # корректировка весов по SGD
    # Q = lm * ek + (1 - lm) * Q  # пересчет показателя качества
    # Q_plot.append(Q)


# line_x = list(range(max(x_train[:, 0])))
# line_y = [-x * w[0] / w[1] - w[2] / w[2] for x in line_x]

# x_0 = x_train[y_train == 1]
# x_1 = x_train[y_train == -1]

# plt.scatter(x_0[:, 0], x_0[:, 1], color="red")
# plt.scatter(x_1[:, 0], x_1[:, 1], color="green")
# print(w, Q)
# plt.plot(line_x, line_y, color="blue")
# plt.grid(True)
# plt.show()

plt.plot(Q_plot, color="b", label="Line Chart")
plt.show()

print(w, Q)
