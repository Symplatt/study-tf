"""
Date: 2024-10-26
Description: PyTorch 自动求导
"""

import torch

# 1. 声明参数需要计算梯度
w = torch.tensor(2.0, requires_grad=True)

# 2. 计算损失
loss = (w - 5) ** 2

# 3. 计算损失对参数的梯度
loss.backward()
print(w.grad)  # tensor(-6.)

# 4. 根据梯度更新参数
grad = w.grad
assert grad is not None
with torch.no_grad():
    w -= 0.1 * grad
