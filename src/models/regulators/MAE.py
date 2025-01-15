for i in range(N):
    k = np.random.randint(0, n_train - 1)
    ek = loss(w, x_train[k], y_train[k])
    Q_mean_gradient = recalculate_gradients(k, w, x_train[k], y_train[k])
    w = w - nt * Q_mean_gradient
    Q = lm * ek + (1 - lm) * Q  # експоненциальная  скользящая  средняя
    Q_plot.append(Q)
    if Q <= 0.03:
        break
