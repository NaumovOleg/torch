import numpy as np


# Пример: квадратичная функция в многомерном пространстве
def loss_function(theta):
    return np.sum((theta - np.array([2, 3])) ** 2)


# Градиент (вектор частных производных)
def gradient(theta):
    return 2 * (theta - np.array([2, 3]))


theta = np.array([5.0, 7.0])  # начальное значение
grad = gradient(theta)

arr = np.array([[1, 2, 34, 4], [3, 5, 1, 2]])

print(arr.T)
