"""tensors.py"""

import numpy
import torch


tensor = torch.empty(
    3,
    5,
    2,
)
tensor2 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
tensor3 = torch.IntTensor([[1, 2, 3], [4, 5, 6]])

dim = tensor.dim()
size = tensor.size()

numpy_arr = numpy.array([[1, 2, 3], [4, 5, 6]])

# mutable numpy
tensor4 = torch.from_numpy(numpy_arr)

# immutable numpy_arr
tensor5 = torch.tensor(numpy_arr, dtype=torch.float16)

tensor5[0, 0] = 100
tensor6 = tensor5.numpy().copy()
# tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])
tensor7 = torch.zeros(3, 4)
# tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0.],
#         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0.],
#         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0.]])
tensor8 = torch.eye(3, 20)
# tensor([[20, 20, 20, 20, 20],
#         [20, 20, 20, 20, 20],
#         [20, 20, 20, 20, 20]])
tensor9 = torch.full((3, 5), 20)
# tensor([-1, 2, 5, 8, 11, 14, 17])
tensor10 = torch.arange(-1, 20, 3)
# tensor([1, 7, 13, 20], dtype=torch.uint8)
tensor11 = torch.linspace(1, 20, 4, dtype=torch.uint8)
# tensor([[0.5874, 0.3008, 0.0278, 0.0352],
#         [0.5474, 0.7002, 0.4932, 0.3657],
#         [0.1050, 0.4326, 0.9126, 0.4268]], dtype=torch.float16)
tensor12 = torch.rand(3, 4, dtype=torch.float16)

# torch.manual_seed(10)
tensor12 = torch.randn(10, 4, dtype=torch.float16)

tensor12.fill_(100)
tensor12.random_(3, 9)
tensor12.uniform_(1, 2)
tensor12.normal_(0, 1)

tensor13 = torch.Tensor(3, 3, 3, 3).normal_(1, 40)

tensor14 = torch.arange(1, 28).view(3, 9)
tensor14.resize_(2, 3, 3)
tensor14.ravel()
tensor14.permute(1, 2, 0)
tensor15 = torch.arange(32).view(8, 2, 2)
tensor16 = torch.unsqueeze(tensor15, dim=3)
tensor16.squeeze_(dim=3)


print(torch.rand(3, 4, dtype=torch.float16))
