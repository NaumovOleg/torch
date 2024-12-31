# квадратичная  функция  потерь
import numpy as np
import matplotlib.pyplot as plt
from datasets import x_train, y_train

x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)))
y_train = np.array(y_train)

pt = np.sum([x * y for x, y in zip(x_train, y_train)], axis=0)

# np.outer = vector * vector
# Compute outer product np.outer([1, 2, 3], [4, 5]) = [[ 4  5],
#  [ 8 10],
#  [12 15]]
xxt = np.sum([np.outer(x, x) for x in x_train], axis=0)

# np.dot = scalar of  2 vectors
w = np.dot(pt, np.linalg.inv(xxt))

line_x = list(range(max(x_train[:, 0])))
line_y = [-x * w[0] / w[1] - w[2] for x in line_x]
x_o = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

plt.scatter(x_o[:, 0], x_o[:, 1], color="red")
plt.scatter(x_1[:, 0], x_1[:, 1], color="green")

plt.plot(line_x, line_y, color="blue")

plt.show()


print(w)
