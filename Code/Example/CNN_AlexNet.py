"""
012年，AlexNet横空出世。这个模型的名字来源于论文第一作者的姓名Alex Krizhevsky [1]。AlexNet使用了8层卷积神经网络，
并以很大的优势赢得了ImageNet 2012图像识别挑战赛。
它首次证明了学习到的特征可以超越手工设计的特征，从而一举打破计算机视觉研究的前状。
AlexNet与LeNet的设计理念非常相似，但也有显著的区别。

本章实现了简化版的代码
"""

import torch

from torch import nn

from Code.Utils.train import train
from Code.Utils.load_data import get_data_fashion_mnist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            # in_channel, out_channel, kernel_size, stride
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 减小卷积窗口，使用填充为2，保证输入输出形状一致
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，进一步增大了通道数
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 这里使用FashionMnist，所以就不将输出搞成1000了
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        features = self.conv(img)
        out = self.fc(features.view(img.shape[0], -1))
        return out


if __name__ == '__main__':
    net = AlexNet()
    batch_size = 256
    lr, num_epoch = 0.01, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_iter, test_iter = get_data_fashion_mnist(batch_size, resize=224)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epoch)

