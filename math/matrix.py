"""test"""

import torch as t


m1 = t.tensor([[1, 2], [3, 4]])

m2 = t.tensor([[5, 6], [7, 8]])


m3 = t.tensor([[1, 0], [0, 1]])
m4 = t.tensor([[1], [2]])
res = t.matmul(m1, m3)
res = t.matmul(m1, m4)
res = t.matmul(m1, m2)
m5 = t.tensor([[1, 2], [0, 1]], dtype=t.float32)
inverse_matrix = t.linalg.inv(m5)

m1f = t.tensor([[1, 2], [3, 4]], dtype=t.float32)
m2f = t.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ],
    dtype=t.float32,
)

# 1*4 - 2*3
det = t.linalg.det(m1f)

# r = (1 * 5 * 9 + 2 * 7 * 6 + 3 * 4 * 8) - (3 * 5 * 7 + 2 * 4 * 9 + 1 * 6 * 8)
det2 = t.linalg.det(m2f)

# tensor([[1.0, 0.0], [0.0, 1.0]])
m6 = t.matmul(inverse_matrix, m5)


print(det2)
