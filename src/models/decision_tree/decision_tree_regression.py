from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(0, np.pi, 0.1).reshape(-1, 1)

y = np.cos(X)

model = DecisionTreeRegressor(max_depth=30)
model.fit(X, y)

y_predicted = model.predict(X)

print(X.reshape(32))

plt.plot(X, y, label="cos")
plt.plot(X, y_predicted, label="cos predicted")

plt.show()
