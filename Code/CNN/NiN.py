import torch


def NiN_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        torch.nn.ReLU(),
        torch.nn.Conv2d(out_channels, out_channels, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(out_channels, out_channels, 1),
        torch.nn.ReLU()
    )
    return blk


