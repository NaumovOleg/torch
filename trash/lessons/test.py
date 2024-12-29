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
invers_matrix = t.linalg.inv(m5)

# tensor([[1.0, 0.0], [0.0, 1.0]])
m6 = t.matmul(invers_matrix, m5)


print(invers_matrix, m6)
