"""1*1卷积运算"""


import torch


def corr2d_1x1(X, K):
    pipe_input, h, w = X.shape
    pipe_output = K.shape[0]
    # 将高和宽压缩成1维
    X = X.view(pipe_input, h*w)
    # 将卷积层转为适合计算的维度
    K = K.view(pipe_output, pipe_input)
    Y = torch.mm(X, K)
    # 计算结果恢复形状
    return Y.view(pipe_output, h, w)


if __name__ == '__main__':
    x = torch.rand(1, 2, 2)
    k = torch.rand(2, 1, 2, 2)
    print(x)
    print("---")
    print(corr2d_1x1(x, k))