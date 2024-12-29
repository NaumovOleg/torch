import torch

tensor = torch.arange(1, 4)
tensor2 = tensor - 3
tensor3 = tensor**3
tensor4 = -tensor
tensor5 = tensor + tensor3

tensor6 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
tensor7 = tensor6 * tensor3

tensor8 = torch.Tensor([[1, 2, 3, 10, 20, 30, 100, 200, 300]])
t_sum = tensor8.sum()
t_mean = tensor8.mean()
t_max = tensor8.max()

tensor9 = tensor8.view(3, 3)
tensor9.sum(dim=1)
tensor9.max()
tensor9.amax()
tensor9.log10()
tensor9.abs()
tensor9.sin()

tensor10 = torch.vstack([tensor6, tensor7])
torch.corrcoef(tensor10)

print(tensor9, end="\n\n")
print(torch.corrcoef(tensor10))
