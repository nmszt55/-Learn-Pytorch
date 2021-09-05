from Code.CNN.Conv2D import *


# 模拟图像
x = torch.ones((4, 6))
x[:, 2:4] = 0

# 构造一个核