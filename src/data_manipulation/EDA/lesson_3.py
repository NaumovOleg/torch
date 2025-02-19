import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


titanic = pd.read_csv("datasets/raw/train.csv")

survival_counts = titanic.groupby(["Pclass", "Survived"]).size()
survival_counts_sex = titanic.groupby(["Sex", "Pclass", "Survived"]).size()


classes = sorted(titanic["Pclass"].unique())
x = np.arange(len(classes))
width = 0.4

survived_counts = np.array(survival_counts.loc[:, 1].values)
not_survived_counts = np.array(survival_counts.loc[:, 0].values)

print(survival_counts_sex)

female_survived_counts = np.array(survival_counts_sex.female.loc[:, 1])
female_not_survived_counts = np.array(survival_counts_sex.female.loc[:, 0])

male_survived_counts = np.array(survival_counts_sex.male.loc[:, 1])
male_not_survived_counts = np.array(survival_counts_sex.male.loc[:, 0])


# fig, ax = plt.subplots(1, 2)
# ax[0].bar(x + width / 1, female_survived_counts, width, label="survived", color="blue")
# ax[0].bar(
#     x - width / 2, female_not_survived_counts, width, label="not survived", color="red"
# )
# ax[0].set_xlabel("Classes female")
# ax[0].set_ylabel("Count")

# ax[1].bar(x + width / 1, male_survived_counts, width, label="survived", color="blue")
# ax[1].bar(
#     x - width / 2, male_not_survived_counts, width, label="not survived", color="red"
# )
# ax[1].set_xlabel("Classes male")
# ax[1].set_ylabel("Count")
# ax[1].set_xticklabels(classes)
# ax[0].set_xticklabels(classes)
# plt.show()
pclass_abs = pd.crosstab(index=titanic.Pclass, columns=titanic.Survived, normalize=True)
print(pclass_abs)
