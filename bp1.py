# TODO：画出计算图

import torch

eta = 0.05

x = torch.tensor(3.0)
y = torch.tensor(7.0)

w = torch.tensor(2.0, requires_grad=True)

y_pred = w * x

loss = (y_pred - y) ** 2
print(f"loss = {loss}")

loss.backward()

assert w.grad is not None
print(f"w.grad = {w.grad.item()}")

w_old = w.item()
with torch.no_grad():
    w -= eta * w.grad
    print(f"下一个step的w = {w_old} - {w_old} * {eta} = {w.item()}")
