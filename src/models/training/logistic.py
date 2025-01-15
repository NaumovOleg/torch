from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

X, y, *_ = make_classification(
    n_samples=25,
    n_features=1,
    n_informative=1,
    n_redundant=0,
    random_state=11,
    n_clusters_per_class=1,
    class_sep=0.4,
)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(X.mean(), X.std())

x = np.linspace(-3, 3)

model = LogisticRegression()
model.fit(X, y)

predict = np.vectorize(lambda x: model.coef_[0] * x + model.intercept_)

y_predicted = model.coef_[0] * X + model.intercept_
y_predicted = predict(X)

sigmoid = lambda x: 1 / (1 + np.exp(-x))


plt.figure(figsize=(10, 2))
plt.scatter(X, model.predict_proba(X)[:, 1], c=y, s=100, edgecolors="black")
plt.plot(x, sigmoid(model.coef_[0] * x + model.intercept_), color="blue")
plt.yticks(np.arange(0, 2), ["0 class", "1 class"])
plt.ylim(-0.1, 1.1)
plt.xlim(-3.5, 2)
plt.grid(True)
plt.show()
