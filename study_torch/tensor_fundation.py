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
