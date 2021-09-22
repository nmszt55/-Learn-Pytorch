import torch
import torch.nn as nn


from Code.CNN.googlenet_inception import Inception
from Code.Utils.global_avg_pool2d import GlobalAvgPool2D
from Code.Utils.flatten_layer import FlattenLayer


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.b1 = nn.Sequential(
            # 由于使用Fashion-Mnist，这里的输入通道数为1
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                Inception(512, 160, (112, 224), (24, 64), 64),
                                Inception(512, 128, (128, 256), (24, 64), 64),
                                Inception(512, 112, (144, 288), (32, 64), 64),
                                Inception(528, 256, (160, 320), (32, 128), 128),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                Inception(832, 384, (192, 384), (48, 128), 128),
                                GlobalAvgPool2D())

    def forward(self, x):
        net = nn.Sequential(
            self.b1, self.b2, self.b3, self.b4, self.b5,
            FlattenLayer(),
            # 第五层最后的通道数为1024,经过全局平均池化和FlattenLayer,形状为(1024, 1),将此作为线性回归的输入
            # 进行最后的分类任务
            nn.Linear(1024, 10)
        )
        return net(x)


if __name__ == '__main__':
    net = GoogleNet()
    x = torch.rand(1, 1, 96, 96)
    for blk in net.children():
        x = blk(x)
        print(x.shape)
