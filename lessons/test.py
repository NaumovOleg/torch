"""test"""

import torch

# r = [1 * 1 + 2 * 2 + 3 * 3, 4 * 1 + 5 * 2 + 6 * 3, 7 * 1 + 8 * 2 + 9 * 3]
m = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = torch.tensor([1, 2, 3])
mv = torch.matmul(m, v)

# r = [[1 * 5 + 2 * 7, 1 * 6 + 2 * 8], [3 * 5 + 4 * 7, 3 * 6 + 4 * 8]]
A = torch.tensor([[1, 2], [3, 4]])  # Shape (2, 2)
B = torch.tensor([[5, 6], [7, 8]])  # Shape (2, 2)


result = torch.matmul(A, B)
print(result)
