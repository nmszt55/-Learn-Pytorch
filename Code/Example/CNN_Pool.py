"""
使用torch.MaxPool2d实现池化层计算
添加池化层的填充和步幅
"""

import torch

# 4个维度分别代表：批次，维度，高，宽
X = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
print(X)


pool2d_1 = torch.nn.MaxPool2d(3)
print(pool2d_1(X))

pool2d_2 = torch.nn.MaxPool2d((2,3), padding=1, stride=2)
print(pool2d_2(X))

pool2d_3 = torch.nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d_3(X))

X1 = torch.arange(start=5, end=21, dtype=torch.float).view(1, 1, 4, 4)
# 按照通道数进行拼接，组成一个2通道的tensor
X_all = torch.cat([X, X1], dim=1)

pool2d = torch.nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X_all))