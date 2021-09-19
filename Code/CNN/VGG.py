import torch


def vgg_block(num_convs, in_channel, out_channel):
    """
    Param num_convs: 卷基层数
    Param in_channel: 输入通道数
    Param out_channel: 输出通道数
    """
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(torch.nn.Conv2d(in_channel, out_channel, (3, 3), padding=1))
        else:
            blk.append(torch.nn.Conv2d(out_channel, out_channel, (3, 3), padding=1))
        blk.append(torch.nn.ReLU())
    blk.append(torch.nn.MaxPool2d(2, 2))
    return torch.nn.Sequential(*blk)

