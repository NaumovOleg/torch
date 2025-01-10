import numpy as np


ar = np.arange(10).reshape(2, 5)
ar2 = np.expand_dims(ar, axis=0)
ar3 = np.append(ar, [[1, 3, 4, 5, 6]], axis=0)


# a = np.arange(4).reshape(2, 2)
# b = np.linspace(4, 7, 4, dtype=np.int8).reshape(2, 2)

# c = np.hstack((a, b))
# d = np.vstack((a, b))
# e = np.column_stack((a, b))

# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
# a = np.arange(1, 10).reshape(3, 3)
# [[10 11 12]
#  [13 14 15]
#  [16 17 18]]
# b = np.arange(10, 19).reshape(3, 3)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]
#  [13 14 15]
#  [16 17 18]]
# c1 = np.concatenate((a, b), axis=0)

# [1 2 3 4 5 6]
# a = np.r_[[1, 2, 3], [4, 5, 6]]

# [1 2 3 4 5 6 7 8 4 5 6]
# a = np.r_[1:9, [4, 5, 6]]

# [[1 2 3] [7 8 9] [4 5 6]]
# a = np.r_[[[1, 2, 3], [7, 8, 9]], [[4, 5, 6]]]

# [[1]
#  [2]
#  [3]]
# a = np.c_[1:4]

# [[1 4]
#  [2 5]
#  [3 6]]
# a = np.c_[[1, 2, 3], [4, 5, 6]]

# [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
# a = np.hsplit(np.arange(9), 3)
