"""
Date: 2024-10-26
Description: PyTorch 张量基础
"""

import torch

# ————————— 创建张量 —————————

# 指定数据
# a = torch.tensor(data=[1, 2, 3])
a = torch.tensor([1, 2, 3])

# 指定数据2
# b = torch.tensor(data=[[1, 2], [3, 4]])
b = torch.tensor([[1, 2], [3, 4]])

# 创建全零张量
# c = torch.zeros(size=(2, 3))
c = torch.zeros((2, 3))

# 创建全一张量
# d = torch.ones(size=(2, 3))
d = torch.ones((2, 3))

# 创建随机张量，服从正态分布
# e = torch.rand(size=(2, 3))
e = torch.rand((2, 3))

# 创建整数范围，e = tensor([0, 1, 2, 3, 4])
# e = torch.arange(start=0, end=5, step=1)
e = torch.arange(5)

# 创建整数范围张量，步长为3，f = tensor([10, 13, 16, 19])
# f = torch.arange(start=10, end=20, step=3)
f = torch.arange(10, 20, 3)

# 创建随机整数，high 不包含在内
# g = torch.randint(low=0, high=10, size=(3, 4, 5))
g = torch.randint(0, 10, (3, 4, 5))

# ————————— 查看张量属性 —————————

# x = torch.randn(size=(24, 36, 256))
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
print(a @ b)  # 32 (同shape的两个一维向量相乘，会以点积的方式运算，得到一个标量)
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


# ————————— 索引与形状 —————————

x = torch.randn(36, 64, 256)

# 取第0个 batch，结果形状为 [64, 256]
a = x[0]

# 取第0个 batch 中第0个序列位置的全部特征，结果形状为 [256]
b = x[0, 0]

# 取第0个 batch、第0个序列位置、第0个特征，得到0维张量
c = x[0, 0, 0]

# 使用 item() 提取0维张量中的数值，转换为 Python 数值
d = x[0, 0, 0].item()

# 取所有 batch 中第0个序列位置的第0个特征，结果形状为 [36]
e = x[:, 0, 0]

# 取第0个 batch 中所有序列位置的第0个特征，结果形状为 [64]
f = x[0, :, 0]

# 取所有 batch 中第0个序列位置的全部特征，结果形状为 [36, 256]
g = x[:, 0, :]

# 取所有 batch 中第0个序列位置的全部特征，并保留序列维度，结果形状为 [36, 1, 256]
h = x[:, 0:1, :]
