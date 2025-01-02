import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * x - 5 * x + 5


dx = lambda x: 2 * x - 5

N = 50  # number of iterations
xx = 0  # initial value
lr = 0.05  # learning rate


x_plt = np.arange(0, 5.0, 0.01)
f_plt = [f(x) for x in x_plt]
plt.ion()  # interactive mode on
fig, ax = plt.subplots()  # create figure and axis
ax.grid(True)  # show grid


ax.plot(x_plt, f_plt)
point = ax.scatter(xx, f(xx), c="red")


for i in range(N):
    # xx = xx - lr * dx(xx)
    lr = 1 / min(1 + i, N)
    xx = xx - lr * np.sign(dx(xx))
    point.set_offsets([xx, f(xx)])
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

plt.ioff()
ax.scatter(xx, f(xx), c="green")
print("----->", xx)
plt.show()
