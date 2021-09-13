import torch
from torch import nn, optim

from Code.Utils.train import train
from Code.Utils.load_data import get_data_fashion_mnist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义卷积层
        self.conv = nn.Sequential(
            # in_channels, out_channels, kernel_size
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        # 定义线性层（全连接层）
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


if __name__ == '__main__':
    net = LeNet()
    batch_size = 256
    train_iter, test_iter = get_data_fashion_mnist(batch_size)
    lr = 0.01
    num_epoch = 5
    optimizer = optim.Adam(net.parameters(), lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epoch)
