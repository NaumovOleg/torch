from matplotlib import gridspec
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

titanic = pd.read_csv("datasets/raw/train.csv")

# print(titanic.sample(5))
print(titanic.info())

titanic.drop(labels="Cabin", axis=1, inplace=True)
# заполним пропуски в столбце Age медианным значением
titanic.Age.fillna(titanic.Age.median())
# два пропущенных значения в столбце Embarked заполним портом Southhampton
titanic.Embarked.fillna("S")
# проверим результат (найдем общее количество пропусков сначала по столбцам, затем по строкам)

tips = pd.read_csv("datasets/raw/tips.csv")

unique = np.unique(titanic.Survived, return_counts=True)
unique_class = np.unique(titanic.Pclass, return_counts=True)
# print(titanic.describe())

# plt.bar(titanic.Survived.unique(), titanic.Survived.value_counts(normalize=True))
# plt.show()

# print(tips.describe())
plt.boxplot(tips.total_bill, vert=False)
plt.show()

# print(tips.total_bill.plot.box())

# fig, (ax_box, ax_hist) = plt.subplots(
#     2,  # две строки в сетке подграфиков,
#     sharex=True,  # единая шкала по оси x и
#     gridspec_kw={"height_ratios": (0.15, 0.85)},
# )  # пропорция 15/85 по высоте

# # затем создадим графики, указав через параметр ax, в какой подграфик поместить каждый из них
# sns.boxplot(x=tips["total_bill"], ax=ax_box)
# sns.histplot(x=tips["total_bill"], ax=ax_hist, bins=10, kde=True)


# # выведем результат
# plt.show()
