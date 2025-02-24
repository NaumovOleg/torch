from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X = np.array([[1], [2], [3], [4], [5]])

print(scaler.fit_transform(X))
