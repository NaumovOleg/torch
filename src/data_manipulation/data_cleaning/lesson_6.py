import pickle as pk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

titanic = pd.read_csv("datasets/raw/train.csv")

label_encoder = LabelEncoder()
scaler = StandardScaler()
knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")

# pk.dump(label_encoder, open("label_encoder.pk", "wb"))

titanic.Cabin = label_encoder.fit_transform(titanic.Cabin)
titanic.Sex = titanic.Sex.map({"male": 0, "female": 1})
int_columns = titanic.select_dtypes(include=["int64", "float"]).columns
int_titanic = titanic[int_columns].drop(columns=["PassengerId"])
int_titanic_scaled = pd.DataFrame(
    scaler.fit_transform(int_titanic), columns=int_titanic.columns
)


# titanic.Sex = label_encoder.fit_transform(titanic.Sex)
# titanic.Embarked = label_encoder.fit_transform(titanic.Embarked)

titanic_filled = pd.DataFrame(
    knn_imputer.fit_transform(int_titanic_scaled),
    columns=int_titanic_scaled.columns,
)

unscaled = pd.DataFrame(
    scaler.inverse_transform(titanic_filled), columns=titanic_filled.columns
)

titanic = pd.merge(
    titanic.drop(columns=unscaled.columns),
    unscaled,
    left_index=True,
    right_index=True,
)

print(titanic.info(), end="\n\n")
# print(titanic_filled.info(), end="\n\n")

# titanic.to_json("./train.json", orient="records")
