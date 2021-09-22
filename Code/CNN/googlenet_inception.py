import torch
import torch.nn as nn

from torch.nn.functional import relu


class Inception(nn.Module):
    def __init__(self, in_channel, channel_1: int, channel_2: tuple, channel_3: tuple, channel_4: int):
        super(Inception, self).__init__()
        # 线路1
        self.net1 = nn.Conv2d(in_channel, channel_1, 1)

        # 线路2
        self.net2_1 = nn.Conv2d(in_channel, channel_2[0], 1)
        self.net2_2 = nn.Conv2d(channel_2[0], channel_2[1], 3, padding=1)

        # 线路3
        self.net3_1 = nn.Conv2d(in_channel, channel_3[0], 1)
        self.net3_2 = nn.Conv2d(channel_3[0], channel_3[1], 5, padding=2)

        # 线路4
        self.net4_1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.net4_2 = nn.Conv2d(in_channel, channel_4, kernel_size=1)

    def forward(self, x):
        o1 = relu(self.net1(x))
        o2 = relu(self.net2_2(relu(self.net2_1(x))))
        o3 = relu(self.net3_2(relu(self.net3_1(x))))
        o4 = relu(self.net4_2(self.net4_1(x)))
        return torch.cat((o1, o2, o3, o4), dim=1)

