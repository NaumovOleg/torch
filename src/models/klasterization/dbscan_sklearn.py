from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt


X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X2, *_ = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)


model = DBSCAN(eps=1.5, min_samples=5)
labels = model.fit_predict(X2)

plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="viridis", edgecolors="k")
plt.title("DBSCAN Clustering")
plt.show()
