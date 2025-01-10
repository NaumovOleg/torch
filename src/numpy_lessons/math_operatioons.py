import numpy as np

a = np.array([1, 2, 3, 10, 20, 30])

a.sum()  # 66
a.mean()  # 11.0
a.max()  # 30

# [[ 1  2]
#  [ 3 10]
#  [20 30]]
a.resize(3, 2)
a.sum(axis=1)  # [ 3 13 50]
a.sum(axis=0)  # [24 42]
np.abs([-10])  # 10
# [[ 0.84147098  0.90929743]
#  [ 0.14112001 -0.54402111]
#  [ 0.91294525 -0.98803162]]
np.sin(a)

# [[0.10387287 0.80211414 0.00639425]
#  [0.22350406 0.58829773 0.58320345]]
rand = np.random.rand(2, 3)
# np.random.seed(1) - same realisations  of  numbers

# [[20 30]
#  [ 3 10]
#  [ 1  2]]
shuffle = np.random.shuffle(a)


a = np.array([1, 2, 3, 10, 20, 30])
median = np.median(a)  # 6.5  медиана
var = np.var(a)  # 114.66666666666667  дисперсия
std = np.std(a)  # 10.708252269472673  среднеквадратическое отклонение


# [[ 1  2  3]
#  [10 20 30]]
a1 = a.reshape(2, 3)
# [[ 1  2]
#  [ 3 10]
#  [20 30]]
a2 = a.reshape(3, 2)

# Внутреннее произведение
# [[  67  112]
#  [ 670 1120]]
a3 = np.dot(a1, a2)  # Матричное умножение
a3 = np.matmul(a1, a2)  # Матричное умножение

# Внешнее произведение
# [[ 4  5  6]
#  [ 8 10 12]
#  [12 15 18]]
# outer = np.outer([1, 2, 3], [4, 5, 6])

linalg_matrix_rank = np.linalg.matrix_rank(a2)  # 2


# решение линейных уравнений
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([2, 4, 1])

# [ 2.25179981e+16 -4.50359963e+16  2.25179981e+16]
linalg_solve = np.linalg.solve(a, [2, 4, 1])
# [ 2.25179981e+16 -4.50359963e+16  2.25179981e+16]
inv_matr = np.linalg.inv(a) @ y


a = np.array([1, 3, 2, 3, 2, 1, 1, 1, 4, 45])
b = np.array([11, 33, 4, 45])

# (array([ 1,  2,  3,  4, 45]), array([4, 2, 2, 1, 1]))
unique, counts = np.unique(a, return_counts=True)


intersect = np.intersect1d(a, b)  # [ 4 45] same
union = np.union1d(a, b)  # [ 1  2  3  4 11 33 45] unique


print(union)
