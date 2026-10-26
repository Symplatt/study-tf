"""
Date: 2024-10-26
Description: study_torch/tensor_foundation_2.py
"""

from typing import Any

import torch
import torch.nn as nn

# 一、nn.Linear：线性变换层

# 基础语法
# weight = torch.randn(256, 512)
# bias = torch.zeros(512)
# output = x @ weight + bias

# nn写法
weight = torch.randn(256, 512)
linear = nn.Linear(256, 512)
output = linear(weight)
print(output.shape)  # [24, 36, 512]


# 二、nn.Parameter：可训练参数
weight = nn.Parameter(torch.randn(12, 64, 256))


# 三、nn.Module：组织整个模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类 nn.Module 的初始化方法，让 PyTorch 能够正确管理模型中的层和参数。
        self.linear1 = nn.Linear(256, 512)  # 定义FFN中的第一个linear
        self.linear2 = nn.Linear(512, 256)  # 定义FFN中的第二个linear

    # FFN
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
