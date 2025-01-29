# вычисление собственных векторов и собственных чисел

import numpy as np
import matplotlib.pyplot as plt

SIZE = 1000
np.random.seed(123)
x = np.random.normal(size=SIZE)
y = np.random.normal(size=SIZE)
z = (x + y) / 2

F = np.vstack([x, y, z])
FF = 1 / SIZE * F @ F.T
L, W = np.linalg.eig(FF)
WW = sorted(zip(L, W.T), key=lambda lx: lx[0], reverse=True)
WW = np.array([w[1] for w in WW])

print(sorted(L, reverse=True))

# =============
X = np.random.randn(100, 2) @ np.array([[2, 1], [1, 3]])  # Коррелированные данные

# 1️⃣ Центрируем данные (вычитаем среднее)
X_mean = X - np.mean(X, axis=0)

# 2️⃣ Вычисляем ковариационную матрицу
cov_matrix = np.cov(X_mean.T)

# 3️⃣ Находим собственные векторы и собственные значения
eig_values, eig_vectors = np.linalg.eig(cov_matrix)

# 4️⃣ Выбираем главный компонент (с наибольшим собственным значением)
idx = np.argsort(eig_values)[::-1]  # Сортируем по убыванию
eig_vectors = eig_vectors[:, idx]  # Переставляем векторы

# 5️⃣ Проецируем данные на первый главный компонент
X_pca = X_mean @ eig_vectors[:, 0]

# 📊 Визуализация
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Исходные данные")
plt.quiver(
    0,
    0,
    eig_vectors[0, 0],
    eig_vectors[1, 0],
    scale=3,
    color="r",
    label="Первый главный компонент",
)
plt.quiver(
    0,
    0,
    eig_vectors[0, 1],
    eig_vectors[1, 1],
    scale=3,
    color="b",
    label="Второй главный компонент",
)
plt.legend()
plt.show()
