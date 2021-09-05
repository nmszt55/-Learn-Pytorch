import torch


def corr2d(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    :param x: 输入数组
    :param k: 卷积核数组
    :return: 输出一个经过二维互相关运算torch
    """
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i][j] = (x[i:i + h][j: j+w] * k).sum().item()
    return y
