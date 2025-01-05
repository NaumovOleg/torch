beta = 0.1
velocity = np.zeros_like(w)
for i in range(N):
    k = np.random.randint(0, n_train - 1)  # random index
    indexes = np.random.randint(low=0, high=n_train, size=5, dtype=int)

    velocity = beta * velocity - nt * df(w, x_train[k], y_train[k])
    w += velocity

    ek = np.mean([loss(w, x_train[o], y_train[o]) for o in indexes], axis=0)
    Q = lm * ek + (1 - lm) * Q
    Q_plot.append(Q)
    if Q <= 0.022:
        break
