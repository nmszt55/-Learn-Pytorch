"""
平均池化层
"""
import torch
from torch.nn.functional import avg_pool2d


class GlobalAvgPool2D(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2D, self).__init__()

    def forward(self, x):
        # 池化核的大小核输出的大小一致，也就是将所有的值都全局平均为1个数
        return avg_pool2d(x, kernel_size=x.size()[2:])