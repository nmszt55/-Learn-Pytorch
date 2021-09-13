"""
评估准确率api
"""

import torch
from torch import nn


def evaluate_acc(data_iter, net, device=None):
    """评估准确率"""
    if device is None and isinstance(net, nn.Module):
        # 如果没指定设备，就是用net训练使用的device
        device = list(net.parameters())[0].device
    total_sum, count = 0, 0

    # 不计算偏导数，优化性能
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, nn.Module):
                # 开启评估模式，此时不会对网络使用正则化惩罚项和随机丢弃隐藏层
                net.eval()
                total_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().item()
            # 自定义模型
            else:
                if "is_training" in net.__code__.co_varnames:
                    # 如果有is_training这个参数，执行下面的命令
                    total_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    total_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            count += X.shape[0]
    return total_sum / count
