import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import (
    power_transform,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
from datasets import sparse_data
from scipy.sparse import csr_matrix


X, y = fetch_california_housing(return_X_y=True, as_frame=True)

boston = pd.read_csv("datasets/raw/boston.csv")
st_scaler = StandardScaler().fit(boston)
boston_transformed = pd.DataFrame(st_scaler.transform(boston), columns=boston.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# boston.hist(bins=15, figsize=(10, 5))
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# sns.histplot(
#     x=(np.array(boston.LSTAT) - np.mean(boston.LSTAT)) / np.std(boston.LSTAT),
#     bins=15,
#     color="green",
#     ax=ax[1],
# )
# sns.histplot(
#     x=power_transform(boston[["LSTAT"]], method="box-cox").flatten(),
#     bins=12,
#     color="orange",
#     ax=ax[2],
# )

pipeline = make_pipeline(StandardScaler(), LinearRegression(), memory="cache")
# pipeline.fit(X_train, y_train)
# predicted = pipeline.predict(X_test)
# score = pipeline.score(X_test, y_test)
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ],
    memory="cache",
)
score = pipeline.fit(X_train, y_train).score(X_test, y_test)


min_max = MinMaxScaler(feature_range=(0, 1))

boston_min_max = pd.DataFrame(min_max.fit_transform(boston), columns=boston.columns)

# plt.show()


sparse_df = pd.DataFrame(sparse_data)
scaled_sparse_df = pd.DataFrame(
    MinMaxScaler().fit_transform(sparse_df), columns=sparse_df.columns
)
sparse_csr = csr_matrix(sparse_df)

print(sparse_df, end="\n\n")
print(scaled_sparse_df, end="\n\n")
print(sparse_csr, end="\n\n")
