import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)  # [5, 7, 9]
print(a * b)  # [4, 10, 18]
print(a @ b)  # 32
