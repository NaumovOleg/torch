import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


tips = pd.read_csv("datasets/raw/tips.csv")


# sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg")
# plt.title("total_bill vs. tip by time")

sns.heatmap(
    tips[["total_bill", "tip"]].corr(),
    # дополнительно пропишем цветовую гамму
    cmap="coolwarm",
    # и зададим диапазон от -1 до 1
    vmin=-1,
    vmax=1,
)

plt.show()
