# Пример работы L2-регуляризатора

import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 10.1, 0.1)

polynom_function = lambda x: x**3 - 10 * x**2 + 3 * x + 500

# функция в виде полинома x^3 - 10x^2 + 3x + 500
y = np.array([polynom_function(a) for a in x])
x_train, y_train = x[::2], y[::2]

N = 13  # размер признакового пространства (степень полинома N-1)
L = 1  # при увеличении N увеличивается L (кратно): 12; 0.2   13; 20    15; 5000

X = np.array([[a**n for n in range(N)] for a in x])  # матрица входных векторов

IL = np.array(
    [[L if i == j else 0 for j in range(N)] for i in range(N)]
)  # матрица lambda*I


IL[0][0] = 0  # первый коэффициент не регуляризуем
print(IL)
X_train = X[::2]  # обучающая выборка
Y = y_train  # обучающая выборка

# вычисление коэффициентов по формуле w = (XT*X + lambda*I)^-1 * XT * Y
A = np.linalg.inv(X_train.T @ X_train + IL)
w = A @ X_train.T @ Y
print(w)

# отображение исходного графика и прогноза
yy = [np.dot(w, x) for x in X]
plt.plot(x, yy)
plt.plot(x, y)
plt.grid(True)
plt.show()
