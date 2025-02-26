import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.ensemble import IsolationForest

boston = pd.read_csv("datasets/raw/boston.csv")

sns.boxplot(x=boston.RM)
# sns.scatterplot(x=boston.RM, y=boston.MEDV)
z_score = pd.DataFrame(stats.zscore(boston), columns=boston.columns)

more_3_sko = boston[(np.abs(z_score) > 3).any(axis=1)]

print(boston.info())

cleared = boston[~((np.abs(z_score) > 3).any(axis=1))]
print(cleared.info())

# IQR
q1 = boston.RM.quantile(0.25)
q3 = boston.RM.quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

iqr_outliers = boston[(boston.RM < lower_bound) | (boston.RM > upper_bound)].head()
# print(iqr_outliers)

X_boston = boston.drop(columns="MEDV")
y_boston = boston.MEDV

clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_boston)

# создадим столбец с anomaly_score
boston["scores"] = clf.decision_function(X_boston)
# и результатом (выброс (-1) или нет (1))
boston["anomaly"] = clf.predict(X_boston)


X_boston = boston.drop(columns="MEDV")
y_boston = boston.MEDV

clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_boston)

# создадим столбец с anomaly_score
boston["scores"] = clf.decision_function(X_boston)
# и результатом (выброс (-1) или нет (1))
boston["anomaly"] = clf.predict(X_boston)

# посмотрим на количество выбросов
boston[boston.anomaly == -1].shape[0]
