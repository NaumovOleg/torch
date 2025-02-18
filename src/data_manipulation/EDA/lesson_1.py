import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np

from datasets import satisfaction_data, cars_data

cars = pd.DataFrame(cars_data)

satisfaction = pd.DataFrame(satisfaction_data)

data = np.random.default_rng(42).poisson(lam=10, size=100)
unique, counts = np.unique(data, return_counts=True)

calc = np.round(len(data[(data > 6) & (data < 10)]) / len(data), 3)


x = np.arange(15)
# передадим их в функцию poisson.pmf()
# mu в данном случае это матожидание (lambda из формулы)
f = poisson.pmf(x, mu=3)

print(f)


# plt.figure(figsize=(10, 6))
# plt.bar([str(x) for x in x], f, width=0.95, color="green")
# plt.title("Теоретическое распределение количества звонков в минуту", fontsize=16)
# plt.xlabel("количество звонков в минуту", fontsize=16)
# plt.ylabel("относительная частота", fontsize=16)
# plt.show()

x = np.arange(-5, 5, 0.1)

# построим график синусоиды
plt.plot(x, np.sin(x))

# зададим заголовок, подписи к осям и сетку
# plt.title("sin(x)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# plt.show()

cars_plot = cars.plot()
