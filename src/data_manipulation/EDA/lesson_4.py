import pandas as pd

# import matplotlib.pyplot as plt
import numpy as np


tips = pd.read_csv("datasets/raw/tips.csv")
# plt.hist(x=titanic[titanic["Survived"] == 0]["Age"], density=True)
# plt.hist(x=titanic[titanic["Survived"] == 1]["Age"], density=True)
# plt.show()


x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)

# сделаем эту последовательность значениями по оси x,
# а по оси y выведем функцию косинуса
# plt.plot(x, np.cos(x))
# plt.title("cos(x)")
plt.scatter(tips.total_bill, tips.tip)
plt.show()
