import torch


# 小批量随机梯度下降
def sgd(params, lr, batch_size):
    params.data -= lr * params.grad / batch_size


