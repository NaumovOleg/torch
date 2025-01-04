import numpy as np
import matplotlib.pyplot as plt

from datasets import x_train, y_train, x_test

x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)))
x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1), dtype=x_test.dtype)))


# sigmoid loss function
def loss(weights, x, y):
    M = np.dot(weights, x) * y
    return 2 / (1 + np.exp(M))


def df(weights, x, y):
    M = np.dot(weights, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y


n_train = len(x_train)
w = [0.0, 0.0, 0.0]  # initial value of  weights
nt = 0.0005  # шаг сходимости SGD

N = 5000
# lm = 2 / (N + 1)  # 0.01
lm = 0.0001  # скорость "забывания" для Q

Q = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])  # показатель качества
QSet_for_SAG = [df(w, x, y) for x, y in zip(x_train, y_train)]
Q_MEAN_SAG = np.mean([df(w, x, y) for x, y in zip(x_train, y_train)], axis=0)

Q_plot = [Q]


def calculate_sag(index, q_mean_sag, weights, x, y):
    new_sag = df(weights, x, y)
    return q_mean_sag - (QSet_for_SAG[index] / N) + (new_sag / N)


# O = 0
# for i in range(N):
#     O += 1
#     k = np.random.randint(0, n_train - 1)  # random index
#     # learning_coefficient = np.exp(-(i / N))
#     indexes = np.random.randint(low=0, high=n_train, size=7, dtype=int)
#     ek = np.mean([loss(w, x_train[o], y_train[o]) for o in indexes], axis=0)
#     # ek = loss(w, x_train[k], y_train[k])

#     # Q_MEAN_SAG = calculate_sag(k, Q_MEAN_SAG, w, x_train[k], y_train[k])
#     # print("====", Q_MEAN_SAG, "----", df(w, x_train[k], y_train[k]), end="\n")

#     # w = w - nt * Q_MEAN_SAG  # Псевдо градиентный алгоритм SGD
#     w = w - nt * (df(w, x_train[k], y_train[k]))  # Псевдо градиентный алгоритм SGD
#     Q = lm * ek + (1 - lm) * Q  # скользящее экспоненциальное сглаживание
#     Q_plot.append(Q)
#     if Q <= 0.003:
#         break


# L2 regulations ----------------->
L = 3
IL = np.array(
    [[L if i == j else 0 for j in range(len(w))] for i in range(len(w))]
)  # матрица lambda*I
IL[0][0] = 0  # первый коэффициент не регуляризуем
print(IL)

A = np.linalg.inv(x_train.T @ x_train + IL)
w2 = A @ x_train.T @ y_train

# метод импульсов для задачи бинарной классификации
beta = 0.1
velocity = np.zeros_like(w)
for i in range(N):
    k = np.random.randint(0, n_train - 1)  # random index
    indexes = np.random.randint(low=0, high=n_train, size=5, dtype=int)

    velocity = beta * velocity - nt * (df(w, x_train[k], y_train[k]) 
    w += velocity

    ek = np.mean([loss(w, x_train[o], y_train[o]) for o in indexes], axis=0)
    Q = lm * ek + (1 - lm) * Q
    Q_plot.append(Q)
    if Q <= 0.025:
        break


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, weights, threshold=0.5):
    linear_combination = np.dot(X, weights)
    probabilities = sigmoid(linear_combination)
    predictions = (probabilities >= threshold).astype(int)
    return 1 if predictions > 0 else -1


print("Weights ", w, Q)
print("W1 ", np.array([predict(x, w) for x in x_train]))
print("W2 ", np.array([predict(x, w2) for x in x_train]))
print(
    "W3 ",
    np.array([predict(x, w) for x in x_test]),
)
print(
    "W4 ",
    np.array([predict(x, w2) for x in x_test]),
)
print("W: ", y_train)


line_x = list(range(max(x_train[:, 0])))
line_y = [-x * w[0] / w[1] - w[2] / w[2] for x in line_x]
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

# [ 0.6731841  -0.36813167  0.01250469] 0.02999365646718698
# [ 0.67379748 -0.36036042  0.01187883] 0.02993571863767804
# [ 0.64756415 -0.34909803  0.01066069] 0.029981527677636354
# [ 0.72648964 -0.39078496  0.01377667] 0.02212071566958664
# [ 0.05966587 -0.03420555  0.04689331]
