import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics

np.random.seed(42)
height_men = np.round(np.random.normal(180, 10, 1000))
height_women = np.round(np.random.normal(160, 10, 1000))

bins = 15

# plt.hist(height_men, bins=bins, alpha=0.5, label="Men")
# plt.hist(height_women, bins=bins, alpha=0.5, label="Women")

X = np.array(
    [
        1.48,
        1.49,
        1.49,
        1.50,
        1.51,
        1.52,
        1.52,
        1.53,
        1.53,
        1.54,
        1.55,
        1.56,
        1.57,
        1.57,
        1.58,
        1.58,
        1.59,
        1.60,
        1.61,
        1.62,
        1.63,
        1.64,
        1.65,
        1.65,
        1.66,
        1.67,
        1.67,
        1.68,
        1.68,
        1.69,
        1.70,
        1.70,
        1.71,
        1.71,
        1.71,
        1.74,
        1.75,
        1.76,
        1.77,
        1.77,
        1.78,
    ]
)
y = np.array(
    [
        29.1,
        30.0,
        30.1,
        30.2,
        30.4,
        30.6,
        30.8,
        30.9,
        31.0,
        30.6,
        30.7,
        30.9,
        31.0,
        31.2,
        31.3,
        32.0,
        31.4,
        31.9,
        32.4,
        32.8,
        32.8,
        33.3,
        33.6,
        33.0,
        33.9,
        33.8,
        35.0,
        34.5,
        34.7,
        34.6,
        34.2,
        34.8,
        35.5,
        36.0,
        36.2,
        36.3,
        36.6,
        36.8,
        36.8,
        37.0,
        38.5,
    ]
)
# slope2, intercept2 = np.polyfit(X, y, 1)

model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
slope = model.coef_[0]
intercept = model.intercept_

print(slope, intercept)
print("Quality", metrics.mean_squared_error(y, model.predict(X.reshape(-1, 1))))
plt.scatter(X, y, alpha=0.5, label="Women")
plt.plot(
    X,
    slope * X + intercept,
    label="Regression line",
    color="red",
)


plt.show()
