from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import numpy as np

# X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X, *_ = make_blobs(n_samples=1000, centers=4, n_features=20, random_state=42)

x = [
    (89, 151),
    (114, 120),
    (156, 110),
    (163, 153),
    (148, 215),
    (170, 229),
    (319, 166),
    (290, 178),
    (282, 222),
]
x = np.array(x)

model = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolors="k")
plt.title("DBSCAN Clustering")
plt.show()
