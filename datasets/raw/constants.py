import numpy as np


x_train = np.array(
    [
        [10, 50],
        [20, 30],
        [25, 30],
        [22, 60],
        [15, 70],
        [40, 40],
        [30, 45],
        [31, 52],
        [20, 45],
        [41, 30],
        [7, 35],
        [22, 53],
    ]
)
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1])
x_test = np.array(
    [
        [12, 52],
        [21, 31],
        [27, 35],
        [20, 58],
        [17, 72],
        [46, 46],
        [31, 46],
        [31, 52],
        [19, 44],
        [42, 31],
        [10, 38],
        [25, 56],
    ]
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
population = [1400, 97, 67, 144, 45, 12, 59]  # млн. человек
area = [9.6, 0.3, 0.2, 17.1, 2.8, 1.1, 1.2]  # млн. кв. км.
sea = [1] * 5 + [0, 1]  # выход к морю (в этом списке его нет только у Боливии)

# кроме того создадим список кодов стран, которые станут индексом датафрейма
custom_index = ["CN", "VN", "GB", "RU", "AR", "BO", "ZA"]

# создадим пустой словарь
countries_dict = {}

# превратим эти списки в значения словаря,
# одновременно снабдив необходимыми ключами
countries_dict["country"] = country
countries_dict["capital"] = capital
countries_dict["population"] = population
countries_dict["area"] = area
countries_dict["sea"] = sea

# создадим датафрейм


satisfaction_data = {
    "level": [
        "Good",
        "Medium",
        "Good",
        "Medium",
        "Bad",
        "Medium",
        "Good",
        "Medium",
        "Medium",
        "Bad",
    ]
}

cars_data = {
    "model": ["Renault", "Hyundai", "KIA", "Toyota"],
    "stock": [12, 36, 28, 32],
}


shop_data = {
    "month": [
        "01/01/2019",
        "01/02/2019",
        "01/03/2019",
        "01/03/2019",
        "01/04/2019",
        "01/05/2019",
        "01/06/2019",
        "01/07/2019",
        "01/08/2019",
        "01/09/2019",
        "01/10/2019",
        "01/11/2019",
        "01/12/2019",
        "01/12/2019",
    ],
    "profit": [
        "1.20$",
        "1.30$",
        "1.25$",
        "1.25$",
        "1.27$",
        "1.11$",
        "1.23$",
        "1.20$",
        "1.31$",
        "1.24$",
        "1.18$",
        "1.17$",
        "1.23$",
        "1.23$",
    ],
    "MoM": [
        0.03,
        -0.02,
        0.01,
        0.02,
        -0.01,
        -0.015,
        0.017,
        0.04,
        0.02,
        0.01,
        0.00,
        -0.01,
        2.00,
        2.00,
    ],
    "high": [
        "Dubai",
        "Paris",
        "singapour",
        "singapour",
        "moscow",
        "Paris",
        "Madrid",
        "moscow",
        "london",
        "london",
        "Moscow",
        "Rome",
        "madrid",
        "madrid",
    ],
}
