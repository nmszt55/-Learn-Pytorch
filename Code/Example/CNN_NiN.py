import torch

from Code.Utils.global_avg_pool2d import GlobalAvgPool2D
from Code.Utils.flatten_layer import FlattenLayer
from Code.CNN.NiN import NiN_block


class NiN(torch.nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        self.net = torch.nn.Sequential(
            # in_channel, out_channel, kernel_size, stride, padding
            NiN_block(1, 96, 11, 4, 8),
            # kernel_size, stride
            torch.nn.MaxPool2d(3, 2),

            NiN_block(96, 256, 5, 1, 2),
            torch.nn.MaxPool2d(3, 2),

            NiN_block(256, 384, 3, 1, 1),
            torch.nn.MaxPool2d(3, 2),

            torch.nn.Dropout(0.5),
            NiN_block(384, 10, 3, 1, 1),

            GlobalAvgPool2D(),
            FlattenLayer(),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    from Code.Utils.load_data import get_data_fashion_mnist
    from Code.Utils.train import train
    device = torch.device("cuda")
    net = NiN()
    batch_size = 32
    train_iter, test_iter = get_data_fashion_mnist(batch_size, resize=224)
    lr, num_epoch = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epoch)