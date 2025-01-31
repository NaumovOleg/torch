# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import RocCurveDisplay
# from sklearn.datasets import load_wine

# X, y = load_wine(return_X_y=True)
# y = y == 2  # make binary
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# svc = SVC(random_state=42)
# svc.fit(X_train, y_train)

# svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]  # Наклон (коэффициент при X)
intercept = model.intercept_  # Свободный член (смещение)

# Делаем предсказание
y_pred = model.predict(X)

# Вывод результатов
print(f"Коэффициент наклона: {slope}")
print(f"Свободный член (intercept): {intercept}")

print(f"Предсказанные значения: {model.predict(np.array([12,14,15]).reshape(-1, 1))}")

# График
plt.scatter(X, y, color="blue", label="Данные")
plt.plot(X, y_pred, color="red", label="Линейная регрессия")
plt.legend()
plt.show()
