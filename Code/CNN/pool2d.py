"""
实现池化层
"""


import torch


def pool2d(Y, pool_size, mod="max"):
    """
    :param Y:卷积完的输出
    :param pool_size: 池化层大小
    :param mod: 池化模式，平均(avg)or最大(max)
    """
    Y = Y.float()
    ph, pw = pool_size
    output = torch.zeros(Y.shape[0]-ph+1, Y.shape[1]-pw+1)
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            if mod == "max":
                output[x][y] = Y[x:x+ph, y:y+pw].max()
            elif mod == "avg":
                output[x][y] = Y[x:x+ph, y:y+pw].mean()
    return output


if __name__ == '__main__':
    X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), 'avg'))