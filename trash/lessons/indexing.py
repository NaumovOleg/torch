import torch

tensor1 = torch.arange(0, 140, 5, dtype=torch.float16).view(4, 7)


a = tensor1[0, 0].item()
b = tensor1[1:6:2]
c = tensor1[:, 2:6:2]
tensor1[0, 1:5:2] = torch.Tensor([9.1, 9])

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

tensor2 = tensor1[0, [[0, 2, 1]]]

torch.save(tensor1, "tensor1.pt")

tensor3 = torch.Tensor(arr)
d = tensor3[tensor3 > 5]


print(tensor3[tensor3 > 5])
