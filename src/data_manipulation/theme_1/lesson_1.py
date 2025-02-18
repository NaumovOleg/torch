import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_csv = pd.read_csv("datasets/raw/train.csv")

html_data = pd.read_html(
    "https://en.wikipedia.org/wiki/World_population",
    match="World population milestones in billions",
)

country = np.array(
    [
        "China",
        "Vietnam",
        "United Kingdom",
        "Russia",
        "Argentina",
        "Bolivia",
        "South Africa",
    ]
)
capital = ["Beijing", "Hanoi", "London", "Moscow", "Buenos Aires", "Sucre", "Pretoria"]
population = [1400, 97, 67, 144, 45, 12, 59]
area = [9.6, 0.3, 0.2, 17.1, 2.8, 1.1, 1.2]
sea = [1] * 5 + [0, 1]


dictionary = {
    "country": country,
    "capital": capital,
    "population": population,
    "area": area,
    "sea": sea,
}

custom_index = ["CN", "VN", "GB", "RU", "AR", "BO", "ZA"]
dataframe = pd.DataFrame(dictionary, index=custom_index)
# print(dataframe, end="\n\n")
# print("columns", dataframe.columns, end="\n\n")
# print("index", dataframe.index, end="\n\n")
# print("values", dataframe.values, end="\n\n")
# print("memory_usage", dataframe.memory_usage(), end="\n\n")
# print(dataframe.to_dict(), end="\n\n")
# print(dataframe.to_numpy(), end="\n\n")
# print(dataframe.country.to_list(), end="\n\n")

country_list = [
    "China",
    "South Africa",
    "United Kingdom",
    "Russia",
    "Argentina",
    "Vietnam",
    "Australia",
]

country_Series = pd.Series(country_list)

# dataframe.reset_index(inplace=True)

# print(country_Series, end="\n\n")

# for index, row in dataframe.iterrows():
# print(index, row["capital"])


# print(dataframe[["capital", "area"]])

locs = dataframe.loc[["CN", "RU", "VN"], ["capital", "population", "area"]]

mask = dataframe.population > 1
filtered = dataframe[(dataframe.population > 50) & (dataframe.area < 2)]
queried = dataframe.query("population > 50 and area < 2")
is_in = dataframe[dataframe.country.isin(["China", "South Africa"])]
# .country.str.startswith('A')
# .nlargest(3, 'population')
dataframe.sort_values(by="population", inplace=False, ascending=True)
