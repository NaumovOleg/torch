from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


model = LinearRegression()

X, y, coef = make_regression(
    n_samples=50, n_features=1, n_informative=1, noise=10, random_state=11, coef=True
)


model.fit(X, y)
# model.coef_ #  Угол наклона
# model.intercept_ #  Сдвиг

x = np.arange(-3, 4)

print("Coef", model.coef_, "Intercept", model.intercept_)
y_predicted = model.predict(X)

MSE = np.mean((y_predicted - y) ** 2)
RMSE = np.sqrt(MSE)
print(RMSE, np.max(y_predicted))


print("Predicted", model.predict(np.array(X[:1]).reshape(-1, 1)), end="\n\n")
print("Given", X[:1], model.predict(X[:1]), y[:1])


model_y = model.coef_ * x + model.intercept_

fig = plt.figure()
plt.scatter(X, y)
plt.plot(x, model_y, color="red", label="Regression Line")
plt.show()
