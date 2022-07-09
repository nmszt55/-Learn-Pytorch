import os
import sys

import torch
import torchvision
from Code.Utils.image_augmentation import show_images, plt
from Code.Utils.load_data_cifar10 import load_cifar10
from Code.CNN.res_net import resnet18
from Code.Utils.train import train_img_aug as train

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CIFAR_PATH = os.path.join(ROOT, "Datasets", "cifar-10")
all_imgs = torchvision.datasets.CIFAR10(CIFAR_PATH, train=True, download=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# show_images([all_imgs[i][0] for i in range(32)], 4, 8, scale=0.8)
# plt.show()

filp_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

no_aug = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def train_with_data_aug(train_augs, test_augs, lr=0.01):
    batch_size, net = 256, resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epoch=10)


train_with_data_aug(filp_aug, no_aug)