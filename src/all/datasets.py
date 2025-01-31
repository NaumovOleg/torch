from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target
describe = df.describe().round(3)

is_null = df.isnull().sum()
is_nan = df.isna().sum()
corr = df.corr().round(3)

X = df[["bp", "s5", "s4", "bmi"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

error = np.sqrt(mean_squared_error(y_test, y_pred))
print(error)
