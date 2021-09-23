import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    is_training: 是否训练模式
    X: 输入前向传播计算完后的样本
    gamma:拉伸量
    beta:偏移量
    moving_mean:
    moving_var:
    eps: 偏移量
    """
    if not is_training:
        x_hat = (X - moving_mean)/torch.sqrt(moving_var+eps)
    else:
        # 1. 先计算均值和方差
        assert len(X.shape) in (2, 4)
        # 计算全连接层
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean()
        # 计算卷积层
        else:
            # 合并批次维度，除了通道维，其他维度压缩成平均值
            mean = X.mead(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        x_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    y = gamma * x_hat + beta
    return y, moving_mean, moving_var


class BatchNorm(torch.nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度的参数
        self.gamma = torch.nn.Parameter(torch.ones(shape))
        self.beta = torch.nn.Parameter(torch.zeros(shape))
        # 不参与求梯度的变量
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.To(X.device)
            self.moving_var = self.moving_var.To(X.device)
        # Module默认的training属性为True
        Y, moving_mean, moving_var = batch_norm(self.training, X, self.gamma, self.beta, self.moving_mean,
                                                self.moving_var, 1e-5, momentum=0.9)
        return Y