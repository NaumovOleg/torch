import matplotlib.pyplot as plt
import numpy as np
import statistics as st

np.random.seed(123)

year = ["2020", "2021", "2022"]
quantity = [100, 200, 300]

# plt.bar(year, quantity)
# plt.show()

height = np.array(np.round(np.random.normal(170, 10, 1000)), dtype=int)
# print(height)

bins = 10
# plt.hist(height, bins)
# plt.show()
mean = np.mean(height)
median = np.median(height)
var = np.var(height)
mode = st.mode(height)
std = np.std(height)


print({"mean": mean, "median": median, "var": var, "mode": mode, "std": std})
