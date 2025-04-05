from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors

grid = GridSearchCV(
    NearestNeighbors(), param_grid={"n_neighbors": [1, 2, 3, 4, 5]}, cv=5
)
grid.fit(user_item_matrix)
