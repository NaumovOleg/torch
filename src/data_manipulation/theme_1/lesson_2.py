import pandas as pd
import numpy as np
from datasets.raw.constants import countries_dict, custom_index

countries = pd.DataFrame(countries_dict, index=custom_index)

copy = countries.copy()
copy.drop(labels="AR", axis=0, inplace=True)
copy.rename(columns={"capital": "city"}, inplace=True)
countries.area = countries.area.astype("int")
by_type = countries.select_dtypes(include=["int64", "float64"])
excluded = countries.select_dtypes(exclude=["object", "category"])


dict_ = {
    "country": "Canada",
    "city": "Ottawa",
    "population": 38,
    "area": 10,
    "sea": "1",
}

# countries = countries._append(dict_, ignore_index=True)

list_of_series = [
    pd.Series(["Spain", "Madrid", 47, 0.5, 1], index=countries.columns),
    pd.Series(["Netherlands", "Amsterdam", 17, 0.04, 1], index=countries.columns),
]

# нам по-прежнему необходим параметр ignore_index = True
# df = countries._append(list_of_series, ignore_index=True)
countries.insert(
    loc=1,
    column="code",  # название столбца
    value=["CN", "VN", "GB", "RU", "AR", "ES", "NL"],
)
countries.reset_index(inplace=True)

countries = countries.assign(area_miles=countries.area / 2.59).round(2)
countries.drop(labels="code", axis=1, inplace=True)

countries["area_miles"] = (countries.area / 2).round(2)

countries.drop(labels=[0, 1], axis=0, inplace=True)

print(countries)
countries.drop(index=[3], inplace=True)
countries.reset_index(inplace=True)


countries.drop(index=countries[countries.area > 2].index.to_list(), inplace=True)
# ==============================================================
people = pd.DataFrame(
    {
        "name": ["Алексей", "Иван", "Анна", "Ольга", "Николай"],
        "gender": [1, 1, 0, 2, 1],
        "age": [35, 20, 13, 28, 16],
        "height": [180.46, 182.26, 165.12, 168.04, 178.68],
        "weight": [73.61, 75.34, 50.22, 52.14, 69.72],
    }
)
gender_map = {0: "female", 1: "male"}
people["gender"] = people.gender.map(gender_map)
people["age_group"] = people["age"].map(lambda x: "adult" if x >= 18 else "minor")
people["age_group2"] = np.where(people["age"] >= 18, "adult", "minor")


def get_age_group(age, threshold):
    if age >= int(threshold):
        age_group = "adult"
    else:
        age_group = "minor"
    return age_group


people["age_group3"] = people["age"].apply(get_age_group, threshold=21)
people[["height", "weight"]] = people[["height", "weight"]].apply(np.median, axis=0)


def get_bmi(x):
    bmi = x["weight"] / (x["height"] / 100) ** 2
    return bmi


# для применения функции к строке используется параметр axis = 1
people["bmi"] = people.apply(get_bmi, axis=1).round(2)


s1 = pd.DataFrame({"item": ["карандаш", "ручка", "папка"], "price": [220, 340, 200]})

s2 = pd.DataFrame(
    {"item": ["клей", "корректор", "скрепка", "бумага"], "price": [200, 240, 100, 300]}
)


concated = pd.concat([s1, s2], axis=1)
math_dict = {
    "name": ["Андрей", "Елена", "Антон", "Татьяна"],
    "math_score": [83, 84, 78, 80],
}

math_degree_dict = {"degree": ["B", "M", "B", "M"]}

cs_dict = {
    "name": ["Андрей", "Ольга", "Евгений", "Татьяна"],
    "cs_score": [87, 82, 77, 81],
}

math = pd.DataFrame(math_dict)
cs = pd.DataFrame(cs_dict)
math_degree = pd.DataFrame(math_degree_dict)


print(math, end="\n\n")
print(cs, end="\n\n")
print(math_degree, end="\n\n")

merged = pd.merge(
    math,
    math_degree,  # выполним соединение двух датафреймов
    how="left",  # способом left join
    left_index=True,
    right_index=True,
)

# merged = merged.join(cs, how="left", rsuffix="cs")
# merged = merged.merge(cs, how="right", on="name", indicator=True)
merged = merged.merge(cs, how="left", on="name", indicator=True)
grouped = merged.groupby("degree")

calculated = grouped.agg(
    {
        "math_score": ["mean", "median", "max", "min"],
        "cs_score": ["mean", "median", "max", "min"],
    }
)


def below29(x):
    m = x.mean()
    return True if m < 29 else False


calculated_2 = grouped.cs_score.agg(["mean", "median", "max", "min", below29])


def standartize(x):
    print("++=++++", x)
    if x.isna().sum() > 0:
        return x
    else:
        return (x - x.mean()) / x.std()


standartized = grouped.cs_score.apply(standartize)

filtered = merged.groupby("degree").filter(lambda x: x["cs_score"].mean() >= 26).head()

cars = pd.read_csv("datasets/raw/cars.csv")
cars.drop(columns=["Unnamed: 0", "vin", "lot", "condition"], inplace=True)
# print(cars.head(), end="\n\n")


def custom_mean(x):
    return sum(x) / len(x)


cars = cars.pivot_table(
    index="brand", values=["year", "price"], aggfunc=["median", "mean", custom_mean]
).transpose()

print(cars.head(), end="\n\n")
