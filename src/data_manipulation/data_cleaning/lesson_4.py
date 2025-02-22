import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

scaler = StandardScaler()


def gaussian_noise(x, scale=1, mu=0, random_state=42):
    rnd = np.random.default_rng(random_state)
    return x + rnd.normal(mu, scale, len(x))


titanic_data = pd.read_csv("datasets/raw/train.csv")
titanic = titanic_data.copy()
titanic.Sex = titanic["Sex"].map({"male": 0, "female": 1})
titanic.drop(columns=["PassengerId"], inplace=True)
titanic.dropna(inplace=True, subset=["Embarked"])
number_columns = titanic.select_dtypes(include=["float64", "int64"]).columns
scaled = scaler.fit_transform(titanic[number_columns])
titanic = pd.DataFrame(scaled, columns=number_columns)

train = titanic[~titanic.Age.isna()]
test = titanic[titanic.Age.isna()]

y_train = train.Age
X_train = train.drop("Age", axis=1)
X_test = test.drop("Age", axis=1)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test.Age = y_pred
titanic = pd.concat([train, test])
titanic.sort_index(inplace=True)

titanic = pd.DataFrame(scaler.inverse_transform(titanic), columns=titanic.columns)

titanic.Age = titanic.Age.round(1).clip(lower=0.5)


titanic["Age_type"] = np.where(titanic.index.isin(train.index), "actual", "imputed")
# sns.histplot(titanic.Age.tolist(), bins=20)

# sns.scatterplot(data=titanic, x=titanic.index, y="Age", hue="Age_type")
test["Age"] = test.Age.clip(lower=0.5)
test.Age = gaussian_noise(x=test.Age)
concated = pd.concat([train, test])

titanic = pd.DataFrame(scaler.inverse_transform(concated), columns=concated.columns)
titanic["Age_type"] = np.where(titanic.index.isin(train.index), "actual", "imputed")
sns.scatterplot(data=titanic, x=titanic.index, y="Age", hue="Age_type")


plt.show()
