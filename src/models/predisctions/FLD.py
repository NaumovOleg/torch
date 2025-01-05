# квадратичная  функция  потерь
import numpy as np
import matplotlib.pyplot as plt
from datasets import x_train, y_train

x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)))
pt = np.sum([x * y for x, y in zip(x_train, y_train)], axis=0)

# np.outer = vector * vector
# Compute outer product np.outer([1, 2, 3], [4, 5]) = [[ 4  5],
#  [ 8 10],
#  [12 15]]
xxt = np.sum([np.outer(x, x) for x in x_train], axis=0)

# np.dot = scalar of  2 vectors
w = np.dot(pt, np.linalg.inv(xxt))

line_x = list(range(max(x_train[:, 0])))  # формирование графика разделяющей линии
line_y = [-x * w[0] / w[1] - w[2] / w[1] for x in line_x]

x_o = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

plt.scatter(x_o[:, 0], x_o[:, 1], color="red")
plt.scatter(x_1[:, 0], x_1[:, 1], color="green")
plt.plot(line_x, line_y, color="blue")
plt.show()
print(w)

# ---------------------------------------------------------------
np.random.seed(42)
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)  # Признаки
y = 4 + 3 * X + np.random.randn(n_samples, 1)  # Истинные значения (с шумом)

# Инициализация параметров модели
w = np.random.randn(1, 1)  # Веса
b = np.random.randn(1)  # Смещение
learning_rate = 0.1  # Шаг обучения
n_iterations = 1000  # Количество итераций
m = len(y)  # Количество наблюдений

# Градиентный спуск
for iteration in range(n_iterations):
    # Предсказания модели
    y_pred = X.dot(w) + b

    # Вычисление градиентов
    dw = (2 / m) * X.T.dot(y_pred - y)  # Градиент по w
    db = (2 / m) * np.sum(y_pred - y)  # Градиент по b

    # Обновление параметров
    w -= learning_rate * dw
    b -= learning_rate * db

    # Вывод ошибки каждые 100 итераций
    if iteration % 100 == 0:
        loss = np.mean((y - y_pred) ** 2)  # Среднеквадратичная ошибка
        print(f"Iteration {iteration}, Loss: {loss:.4f}")

# Итоговые параметры
print("\nОптимальные параметры:")
print(f"w: {w[0][0]:.4f}, b: {b[0]:.4f}")

# Проверка на новых данных
X_new = np.array([[0], [2]])  # Новые точки
y_new_pred = X_new.dot(w) + b
print("\nПредсказания для новых данных:")
print(y_new_pred)
