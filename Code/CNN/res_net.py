import torch

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


if __name__ == '__main__':
    from Code.Utils.load_data import get_data_fashion_mnist
    from Code.Utils.train import train
    device = torch.device("cuda")
    train_iter, test_iter = get_data_fashion_mnist(256)
    net = ResNet()
    optimizer = torch.optim.Adam(net.parameters(), 0.01)
    train(net, train_iter, test_iter, 256, optimizer, device, 5)
