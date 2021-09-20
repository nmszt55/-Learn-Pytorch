import torch

from Code.CNN.VGG import vgg_block
from Code.Utils.flatten_layer import FlattenLayer


# 定义5层VGG Block
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
# out_channel * w * h
fc_features = 512 * 7 * 7
# 任意
fc_hidden_units = 4096


def vgg_11(conv_arch, fc_features, fc_hidden_units):
    net = torch.nn.Sequential()
    for i, (num_conv, in_channel, out_channel) in enumerate(conv_arch, start=1):
        # 由于池化层的关系，每次VGG输出的形状，会比输入的形状减小1/2
        net.add_module("VGG_%d" % i, vgg_block(num_conv, in_channel, out_channel))

    net.add_module("fc", torch.nn.Sequential(
        FlattenLayer(),
        # 1
        torch.nn.Linear(fc_features, fc_hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        # 2
        torch.nn.Linear(fc_hidden_units, fc_hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        # 3
        torch.nn.Linear(fc_hidden_units, 10)
    ))
    return net


if __name__ == '__main__':
    device = torch.device("cuda")
    ratio = 8
    conv_arch = ((1, 1, 64//ratio),
                 (1, 64//ratio, 128//ratio),
                 (2, 128//ratio, 256//ratio),
                 (2, 256//ratio, 512//ratio),
                 (2, 512//ratio, 512//ratio))
    net = vgg_11(conv_arch, fc_features//ratio, fc_hidden_units//ratio)
    from Code.Utils.train import train
    from Code.Utils.load_data import get_data_fashion_mnist
    batch_size = 64
    train_iter, test_iter = get_data_fashion_mnist(batch_size, resize=224)
    lr, num_epoch = 0.01, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epoch)
