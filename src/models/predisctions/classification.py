import numpy as np
import matplotlib.pyplot as plt
from datasets import x_train, y_train

n_train = len(x_train)
w = [0.0, -1.0]


def a(x):
    return np.sign(x[0] * w[0] + x[1] * w[1])


N = 50  # number of iterations
L = 0.1  # learning rate
e = 0.1  # learning rate

last_error_index = -1

print("-----,", [1 for i in range(n_train) if y_train[i] * a(x_train[i]) < 0])

for n in range(N):
    for i in range(n_train):
        if y_train[i] * a(x_train[i]) < 0:
            print(
                "---->",
                y_train[i],
                a(x_train[i]),
                y_train[i] * a(x_train[i]),
            )
            w[0] = w[0] + L * y_train[i]
            last_error_index = i
    Q = sum([1 for i in range(n_train) if y_train[i] * a(x_train[i]) < 0])
    if Q == 0:
        break
if last_error_index > -1:
    w[0] = w[0] + e * y_train[last_error_index]

print(w)

line_x = range(max(x_train[:, 0]))
line_y = [w[0] * x for x in line_x]

x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

print(x_0[:, 0], x_0[:, 1])

plt.scatter(x_0[:, 0], x_0[:, 1], color="red")
plt.scatter(x_1[:, 0], x_1[:, 1], color="blue")
plt.plot(line_x, line_y, color="green")

plt.xlim(0, 45)
plt.ylim(0, 75)
plt.show()


print(x_train.shape)
