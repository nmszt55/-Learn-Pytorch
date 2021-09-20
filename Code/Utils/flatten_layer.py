from torch import nn


class FlattenLayer(nn.Module):
    """
    作用是将输入转换为1列，方便后续进行线性运算
    """
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)

