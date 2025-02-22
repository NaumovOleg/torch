from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt

scaler = StandardScaler()
dataframe = pd.read_csv("datasets/raw/train.csv")
columns = dataframe.select_dtypes(include=["float64", "int64"]).columns
titanic = dataframe[columns]

titanic = pd.DataFrame(scaler.fit_transform(titanic), columns=titanic.columns)

titanic_imputer = IterativeImputer(
    initial_strategy="mean",  # вначале заполним пропуски средним значением
    estimator=LinearRegression(),  # в качестве модели используем линейную регрессию
    random_state=42,  # добавим точку отсчета
)

titanic = titanic_imputer.fit_transform(titanic)
titanic = pd.DataFrame(scaler.inverse_transform(titanic), columns=columns)
titanic.Age = titanic.Age.round(1)
titanic.Age = titanic.Age.clip(lower=0.5)

# sns.scatterplot(data=titanic, x=titanic.index, y="Age")
sns.histplot(titanic.Age.tolist(), bins=20)


knn = titanic.copy()
knn = pd.DataFrame(scaler.fit_transform(knn), columns=knn.columns)
knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
knn = pd.DataFrame(knn_imputer.fit_transform(knn), columns=knn.columns)
knn = pd.DataFrame(scaler.inverse_transform(knn), columns=knn.columns)

print(knn)
