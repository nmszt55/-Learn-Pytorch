import torch
import torch.nn as nn

from Code.Utils.global_avg_pool2d import GlobalAvgPool2D
from Code.Utils.flatten_layer import FlattenLayer


def conv_block(in_channel, out_channel):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 3, padding=1)
    )
    return blk


class DenseBlock(nn.Module):
    def __init__(self, num_block, in_channel, out_channel):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_block):
            in_c = in_channel + i*out_channel
            net.append(conv_block(in_c, out_channel))
        self.net = torch.nn.ModuleList(net)
        self.out_channel = in_channel + num_block * out_channel

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x, y), dim=1)
        return x


def transition_block(in_channel, out_channel):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.MaxPool2d(2, 2)
    )
    return blk


class DenseNet(nn.Module):
    def __init__(self, num_channels, growth_rate, num_convs: list):
        """
        num_channels: 输入频道数
        growth_rate: 每个denseblock内部的卷积层输出层数增长率。
        num_convs: 每个DenseBlock内部有多少个卷积层，例如[1,2,3,4],说明第一个block包含1个卷积，第二个包含2个。。。。
        """
        super(DenseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        for i, conv_num in enumerate(num_convs):
            denseblock = DenseBlock(conv_num, num_channels, growth_rate)
            self.net.add_module("dense%d" % i, denseblock)
            num_channels = denseblock.out_channel
            # 添加过渡层
            if i != len(num_convs)-1:
                trans = transition_block(num_channels, num_channels//2)
                self.net.add_module("transition%d" % i, trans)
                num_channels //= 2
        # 接上全局池化和全连接
        self.net.add_module("bn", nn.BatchNorm2d(num_channels))
        self.net.add_module("relu", nn.ReLU()),
        self.net.add_module("global_avg", GlobalAvgPool2D()),
        self.net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    from Code.Utils.train import train
    from Code.Utils.load_data import get_data_fashion_mnist
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DenseNet(64, 32, [4,4,4,4])
    train_iter, test_iter = get_data_fashion_mnist(batch_size, resize=96)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    num_epoch = 5
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epoch)

