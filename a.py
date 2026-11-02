import torch

a = torch.tensor([1.0, 2.0, 3.0])

print(a.sum())  # 6
print(a.mean())  # 2

b = torch.tensor([4, 5, 6], dtype=torch.float32)

print(a + b)  # [5, 7, 9]
print(a * b)  # [4, 10, 18]

c = torch.tensor([4.0, 5.0, 6.0])

print(a + c)
print(a * c)

print(torch.equal(b, c))
print(b.dtype == c.dtype)
print(b.shape == c.shape)
