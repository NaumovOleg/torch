import numpy as np


# Сигмоида
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Функция потерь
def loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def loss_with_l2(y_true, y_pred, weights, lambda_reg):
    m = y_true.shape[0]
    log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    l2_penalty = (lambda_reg / (2 * m)) * np.sum(weights[1:] ** 2)  # Не штрафуем bias
    return log_loss + l2_penalty


def gradient_descent_with_l2(X, y, weights, learning_rate, lambda_reg):
    m = X.shape[0]
    predictions = sigmoid(np.dot(X, weights))
    errors = predictions - y
    gradient = np.dot(X.T, errors) / m
    weights[1:] -= (lambda_reg / m) * weights[
        1:
    ]  # Добавляем L2 только для ненулевых весов
    weights -= learning_rate * gradient
    return weights


# Градиентный спуск
def gradient_descent(X, y, weights, learning_rate):
    m = X.shape[0]
    predictions = sigmoid(np.dot(X, weights))
    errors = predictions - y
    gradient = np.dot(X.T, errors) / m
    weights -= learning_rate * gradient
    return weights


# Функция предсказания
def predict(X, weights, threshold=0.5):
    probabilities = sigmoid(np.dot(X, weights))
    return (probabilities >= threshold).astype(int)


# Данные
X = np.array([[2.5], [3.0], [3.5], [3.8], [4.0], [1.8], [2.8], [3.3], [3.9], [2.1]])
y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])

# Добавляем столбец единиц для bias
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Параметры
weights = np.zeros(X.shape[1])
learning_rate = 0.1
epochs = 10000

# Обучение
for epoch in range(epochs):
    weights = gradient_descent(X, y, weights, learning_rate)
    if epoch % 100 == 0:
        predictions = sigmoid(np.dot(X, weights))
        current_loss = loss(y, predictions)
        print(f"Эпоха {epoch}, Потери: {current_loss:.4f}")

# Прогноз
y_pred = predict(X, weights)
print("Предсказанные классы:", y_pred)
