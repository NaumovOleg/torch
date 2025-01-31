import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
)

scaler = StandardScaler()

iris = load_iris()
data = iris.data
feature_names = iris.feature_names
y = iris.target

# X = scaler.fit_transform(pd.DataFrame(data, columns=feature_names))
X = data
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df["target"] = y

correlation_matrix = iris_df.corr(method="pearson")
# wcc = []
# for i in range(1, 10):
#     kmeans = KMeans(
#         n_clusters=i, init="k-means++", random_state=42, n_init=10, max_iter=300
#     )
#     kmeans.fit(X)
#     wcc.append(kmeans.inertia_)
#     print(kmeans.inertia_)

model = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10, max_iter=300)
model.fit(X)
y_predicted = model.predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 10), wcc, marker="o", linestyle="-", color="b")

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=y_predicted, cmap="viridis", edgecolors="k", s=100
)
# plt.scatter(
#     kmeans.cluster_centers_[:, 0],
#     kmeans.cluster_centers_[:, 1],
#     c="red",
#     marker="X",
#     s=200,
#     label="Centroids",
# )

plt.show()
# plt.show()
y_normalized = np.array(
    [0 if x == 1 else (1 if x == 0 else x) for x in y_predicted], dtype=int
)


print(y, end="\n\n")
print(y_predicted, end="\n\n")
print(y_normalized, end="\n\n")

Q = {
    "r2": r2_score(y, y_normalized),
    "mae": mean_absolute_error(y, y_normalized),
    "mse": mean_squared_error(y, y_normalized),
    "rmse": np.sqrt(mean_squared_error(y, y_normalized)),
}


predicted_df = pd.DataFrame(
    {"Target": y, "Predicted": y_normalized}, columns=["Target", "Predicted"]
)

print(Q)
print(predicted_df)
