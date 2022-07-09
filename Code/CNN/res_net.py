import torch
from torch import nn
from torch.nn import functional as F

from Code.Utils.global_avg_pool2d import GlobalAvgPool2D
from Code.Utils.flatten_layer import FlattenLayer


class Residual(torch.nn.Module):
    def __init__(self, in_channel, out_channel, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channel, out_channel, 3, padding=1),
            torch.nn.BatchNorm2d(out_channel),
        )
        if use_1x1conv:
            self.conv1x1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None
        self.net.add_module(name="relu_final", module=torch.nn.ReLU())

    def forward(self, X):
        Y = self.net(X)
        if self.conv1x1:
            X = self.conv1x1(X)
        return Y + X


class ResBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, num_block, first_block=False):
        """
        num_block: 残差块的数量
        first_block: 是否为第一层网络，如果是的话，需要确定输入通道与输出通道一致
        """
        super(ResBlock, self).__init__()
        if first_block:
            assert in_channel == out_channel
        s = torch.nn.Sequential()
        for i in range(1, num_block+1):
            if i == 1 and not first_block:
                s.add_module("residual_block%d" % i, Residual(in_channel, out_channel, True, 2))
            else:
                s.add_module("residual_block%d" % i, Residual(out_channel, out_channel))
        self.net = s

    def forward(self, X):
        return self.net(X)


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.net.add_module("resblock-1", ResBlock(64, 64, 2, True))
        self.net.add_module("resblock-2", ResBlock(64, 128, 2))
        self.net.add_module("resblock-3", ResBlock(128, 256, 2))
        self.net.add_module("resblock-4", ResBlock(256, 512, 2))
        # 接连全局平均池化后接全连接层
        self.net.add_module("global_avg_pool", GlobalAvgPool2D())
        self.net.add_module("flatten", FlattenLayer())
        self.net.add_module("Linear", torch.nn.Linear(512, 10))

    def forward(self, X):
        return self.net(X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def resnet18(output=10, in_channels=3):
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, output)))
    return net


if __name__ == '__main__':
    from Code.Utils.load_data import get_data_fashion_mnist
    from Code.Utils.train import train
    device = torch.device("cuda")
    train_iter, test_iter = get_data_fashion_mnist(256)
    net = ResNet()
    optimizer = torch.optim.Adam(net.parameters(), 0.01)
    train(net, train_iter, test_iter, 256, optimizer, device, 5)
