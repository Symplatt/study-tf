import torch

# ————————— 创建张量 —————————

# 指定数据
a = torch.tensor([1, 2, 3])

# 指定数据2
b = torch.tensor([[1, 2], [3, 4]])

# 创建全零张量
c = torch.zeros((2, 3))

# 创建全一张量
d = torch.ones((2, 3))

# 创建随机张量，服从正态分布
e = torch.rand((2, 3))

# 创建整数范围，e = tensor([0, 1, 2, 3, 4])
e = torch.arange(5)

# 创建整数范围张量，步长为3，f = tensor([10, 13, 16, 19])
f = torch.arange(10, 20, 3)

# 创建随机整数，high 不包含在内
g = torch.randint(0, 10, (3, 4, 5))

# ————————— 查看张量属性 —————————

x = torch.randn(24, 36, 256)

print("\nx.shape：", x.shape)  # 查看张量的形状
print("\nx.size()：", x.size())  # 查看张量的形状
print("\nx.dim()：", x.dim())  # 查看张量的维度
print("\nx.dtype：", x.dtype)  # 查看张量的数据类型
print("\nx.device：", x.device)  # 查看张量所在的设备
print("\nx.numel()：", x.numel())  # 查看张量的元素个数
print("\nx.is_cuda：", x.is_cuda)  # 查看张量是否在GPU上
print("\n")

# ————————— 基本运算 —————————

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)  # [5, 7, 9]
print(a * b)  # [4, 10, 18]
print(a @ b)  # 32
print(a.sum())  # 6
print(a.mean())  # 2

c = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(c.shape)
d = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(d.shape)
e = c @ d
print(e.shape)

try:
    f = c @ c
    print(f.shape)
except RuntimeError as e:
    print("Error:", e)
    print("Cannot perform matrix multiplication on tensors with incompatible shapes.")
