import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(48)
x = np.linspace(0, 9, 50)

y = np.random.randint(10, size=len(x))

print(len(x), len(y))

# plt.grid()
# plt.plot(x, np.cos(x - 0), color="blue")  # по названию
# plt.plot(x, x + 0, linestyle="solid", linewidth=2)
# plt.plot(x, x + 1, linestyle="dashed", linewidth=2)
# plt.plot(x, x + 2, linestyle="dashdot", linewidth=2)
# plt.plot(x, x + 3, linestyle="dotted", linewidth=2)
# plt.scatter(x[:10], y[:10], c="b", marker="o")
# plt.scatter(x[10:20], y[10:20], c="r", marker="s")
# plt.axis([-2, 12, -1.5, 11])
# plt.show()

# print(np.sin(x))

# plt.figure(figsize=(8, 5))

# # добавим графики синуса и косинуса с подписями к кривым
# plt.plot(x, np.sin(x), label="sin(x)")
# plt.plot(x, np.cos(x), label="cos(x)")

# # выведем легенду (подписи к кривым) с указанием места на графике и размера шрифта
# plt.legend(loc="lower left", prop={"size": 14})

# # добавим пределы шкал по обеим осям,
# plt.axis([-0.5, 10.5, -1.2, 1.2])

# # а также деления осей графика
# plt.xticks(np.arange(11))
# plt.yticks([-1, 0, 1])

# # добавим заголовок и подписи к осям с указанием размера шрифта
# plt.title("Функции y = sin(x) и y = cos(x)", fontsize=18)
# plt.xlabel("x", fontsize=16)
# plt.ylabel("y", fontsize=16)

# # добавим сетку
# plt.grid()

# # выведем результат
# plt.show()


fig = plt.figure(figsize=(10, 5))
(axes1, axes2) = fig.subplots(2, 2)
ax1, ax2 = axes1
ax3, ax4 = axes2
ax1.plot(x, np.sin(x))
ax2.plot(x, np.cos(x))

# ax4.plot(x, np.cos(x))
ax2.set(
    title="y = cos(x)",
    xlabel="x",
    ylabel="y",
    xlim=(-0.5, 10.5),
    ylim=(-1.2, 1.2),
    xticks=(np.arange(11)),
    yticks=[-1, 0, 1],
)
plt.show()
