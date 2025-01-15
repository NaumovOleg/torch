import numpy as np
from datasets import x_train, y_train

w = np.array([0, 0, 0])

N = 500
beta = 0.1
velocity = np.zeros_like(w)
nt = 0.005

for i in range(N):
    k = np.random.randint(0, len(x_train) - 1)  # random index

    velocity = beta * velocity - nt * df(w, x_train[k], y_train[k])
    w -= velocity

    indexes = np.random.randint(low=0, high=n_train, size=5, dtype=int)
    ek = np.mean([loss(w, x_train[o], y_train[o]) for o in indexes], axis=0)
    Q = lm * ek + (1 - lm) * Q
    Q_plot.append(Q)
    if Q <= 0.022:
        break

    # for epoch in range(epochs):
    #     grad = gradient(w, X, y)  # Вычисляем градиент
    #     v = gamma * v + lr * grad  # Обновляем скорость
    #     w -= v  # Обновляем веса
    #     loss = loss_function(w, X, y)  # Вычисляем функцию потерь
    #     loss_history.append(loss)
    #     print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
