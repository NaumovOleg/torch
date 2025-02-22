import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

plt.grid()

titanic = pd.read_csv("datasets/raw/train.csv")
# titanic.Age.astype("int")
isna_sum = titanic.isna().sum()

sns.set()
# msno.bar(titanic, figsize=(10, 6))
# msno.matrix(titanic, figsize=(10, 6))

# plt.show()

# матрица  корелляции пропущенных значений nullity correlation matrix
correlation_matrix = titanic[["Age", "Cabin", "Embarked"]].isnull().corr()
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# msno.heatmap(titanic, figsize=(10, 6))

# titanic.dropna(inplace=True, subset=["Embarked"])
titanic.drop(columns=["Cabin"], inplace=True)
sex_g = titanic.groupby("Sex").count()

# titanic = titanic[["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age", "Embarked"]]
titanic["Sex"] = titanic["Sex"].map({"male": 0, "female": 1})
# titanic.fillna({"Age": 0}, inplace=True)
# titanic.Age.fillna(titanic.Age.median(), inplace=True)
missing_embarked = titanic[titanic.Embarked.isnull()]

sns.histplot(np.array(titanic.Age), bins=20)

titanic.Age.fillna(
    titanic.groupby(["Sex", "Pclass"]).Age.transform("median"),
    inplace=True,
)

sns.histplot(np.array(titanic.Age), bins=20)


imp_most_freq = SimpleImputer(strategy="most_frequent")

titanic.Embarked = imp_most_freq.fit_transform(titanic[["Embarked"]]).ravel()
survived_by_port = titanic.groupby("Embarked").Survived.count()

print(survived_by_port)
