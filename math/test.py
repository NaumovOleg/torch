import torch as t

matrix = t.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
multiply = matrix * 2

matrix2 = t.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_sum = matrix + matrix2
matrix_mult = matrix * matrix2

matrix_3 = t.Tensor([[1, 2], [3, 4]])
w = t.Tensor([1, 2])
matrix_mult_3 = matrix_3 * w
mv = t.matmul(matrix_3, w)

matrix4 = matrix_3 * matrix_3

r = [
    [1 * 1 + 2 * 0, 1 * 2 + 2 * 1],
    [0 * 1 + 1 * 0, 0 * 2 + 1 * 1],
]
matrix_5 = t.matmul(t.tensor([[1, 2], [0, 1]]), t.tensor([[1, 2], [0, 1]]))
matrix_6 = t.matmul(t.tensor([1, 2]), t.tensor([1, 2]))


print(matrix_6, end="\n")
