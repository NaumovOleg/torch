import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

movies = pd.read_csv("datasets/raw/movie/movies.csv")
ratings = pd.read_csv("datasets/raw/movie/ratings.csv")
movies.drop("genres", axis=1, inplace=True)
ratings.drop("timestamp", axis=1, inplace=True)

user_item_matrix = ratings.pivot(index="movieId", columns="userId", values="rating")
user_item_matrix.fillna(0, inplace=True)

# вначале сгруппируем (объединим) пользователей, возьмем только столбец rating
# и посчитаем, сколько было оценок у каждого пользователя
user_votes = ratings.groupby("userId")["rating"].agg("count")
movie_votes = ratings.groupby("movieId")["rating"].agg("count")

print("-----------", user_item_matrix.loc[0:1])

# теперь создадим фильтр (mask)
user_mask = user_votes[user_votes > 10].index
movie_mask = movie_votes[movie_votes > 50].index

# применим фильтры и отберем фильмы с достаточным количеством оценок, а также активных пользователей
user_item_matrix_filtered = user_item_matrix.loc[movie_mask, :]
user_item_matrix_filtered = user_item_matrix_filtered.loc[:, user_mask]

# В датасете по понятным причинам очень много нулей. Такая матрица называется разреженной
# (sparse matrix). Одновременно, если признаков очень много (а у нас их уже довольно много),
# то говорят про данные с высокой размерностью (high-dimensional data). В таком формате алгоритм
# будет долго обсчитывать расстояния между фильмами. Для того чтобы преодолеть эту сложность,
# можно преобразовать данные в формат сжатого хранения строкой (сompressed sparse row, csr).
csr_data = csr_matrix(user_item_matrix_filtered.values)


user_item_matrix_filtered = user_item_matrix_filtered.rename_axis(
    None, axis=1
).reset_index()

knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

recomendations = 10
search_word = "Matrix"

movie_search = movies[movies["title"].str.contains(search_word, case=False)]
movie_id = movie_search.iloc[0]["movieId"]
movie_id_in_matrix = user_item_matrix_filtered[
    user_item_matrix_filtered["movieId"] == movie_id
]

if movie_id_in_matrix.empty:
    print("DataFrame is empty. Stopping execution.")
    exit()

print(movie_id_in_matrix)
movie_id_in_matrix = movie_id_in_matrix.index[0]

distances, indices = knn.kneighbors(
    csr_data[movie_id_in_matrix], n_neighbors=recomendations + 1
)


# print("-----", movie_search.iloc[0]["movieId"], end="\n")
# print("-----", movie_id_in_matrix, end="\n")
# print("-----", csr_data[movie_id_in_matrix], end="\n")
# print("-----", distances, indices[0], end="\n")


indices_distances = list(zip(indices.squeeze(), distances.squeeze()))
sorted_indices_distances = sorted(indices_distances, key=lambda x: x[1], reverse=True)

# print(movies[movies["movieId"].isin(indices.flatten())])
# print(indices_distances, end="\n")
# print("indices_distances", end="\n")
# print(sorted_indices_distances, end="\n")
# print(sorted_indices_distances, end="\n")

ids = np.array(sorted_indices_distances)[:, 0]

print("========", movies[movies["movieId"].isin(ids)], end="\n")
