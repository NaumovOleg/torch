from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from datasets import x_train, y_train

x_train = np.hstack((x_train, np.ones([x_train.shape[0], 1])))

clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

lin_clf = svm.LinearSVC()
lin_clf.fit(x_train, y_train)

v = clf.support_vectors_
w = lin_clf.coef_[0]

print(v, w, clf.coef_)
print(max(x_train[:, 0]))


# line_x = list(range(int(max(x_train[:, 0]))))
line_x = np.arange(0, int(max(x_train[:, 0])), 1)
line_y = [-x * w[0] / w[1] - w[2] for x in line_x]

x_0 = x_train[y_train == 1]  # формирование точек для 1-го
x_1 = x_train[y_train == -1]  # и 2-го классов

plt.scatter(x_0[:, 0], x_0[:, 1], color="red")
plt.scatter(x_1[:, 0], x_1[:, 1], color="blue")
plt.scatter(v[:, 0], v[:, 1], s=70, edgecolor=None, linewidths=0, marker="s")
plt.plot(line_x, line_y, color="green")

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel("длина")
plt.xlabel("ширина")
plt.grid(True)
plt.show()
