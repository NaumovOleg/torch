from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
)


dataset = load_breast_cancer()
scaler = StandardScaler()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=dataset.feature_names)

scaled_df["target"] = dataset.target

unique, counts = np.unique(dataset.target, return_counts=True)
nullable = df.describe().isna().sum()


# print(df.describe().round(2), end="\n\n")
# print(scaled_df.describe().round(2), end="\n\n")

data = scaled_df.groupby("target").mean().T

features = data.index.tolist()
X = scaled_df[features]
y = scaled_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

Q = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "R2": model.score(X_test, y_test),
    "R2Score": r2_score(y_test, y_pred),
    "Accuracy": accuracy_score(y_test, y_pred),
}

conf_matrix = confusion_matrix(y_test, y_pred)

print(Q)
print(conf_matrix)
