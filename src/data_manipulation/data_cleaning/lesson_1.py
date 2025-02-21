import pandas as pd
from datasets import shop_data
import matplotlib.pyplot as plt

financials = pd.DataFrame(shop_data)
print(financials, end="\n\n")
financials = financials[~financials.duplicated(subset="month", keep="first")]
financials.reset_index(drop=True, inplace=True)

financials.drop_duplicates(
    keep="last", subset=["month"], ignore_index=True, inplace=True
)
financials.MoM = financials.MoM.apply(lambda x: x if (x < 1) else x / 100)
financials["profit"] = financials["profit"].str.strip("$").astype("float")
financials["high"] = financials["high"].str.title()
financials["date"] = pd.to_datetime(
    financials["month"], format="%d/%m/%Y", dayfirst=True
)
financials.drop("month", axis=1, inplace=True)
financials.sort_values(by="date", inplace=True)
financials.reset_index(drop=True, inplace=True)
print(financials, end="\n\n")
print(financials.profit.sum(), end="\n\n")

financials[["profit", "MoM"]].plot(
    subplots=True,
    layout=(1, 2),
    kind="line",
)
plt.show()
