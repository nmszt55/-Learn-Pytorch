import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import sys

from Code import DATADIR
from Code.Utils.image_augmentation import show_images, plt
from Code.Utils.train import train_img_aug as train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(DATADIR, "hotdog", "hotdog")

train_imgs = ImageFolder(os.path.join(DATA_DIR, "train"))
test_imgs = ImageFolder(os.path.join(DATA_DIR, "test"))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
# show_images(hotdogs + not_hotdogs, 2, 8, 1.4)
# plt.show()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize])

test_augs = transforms.Compose([transforms.Resize(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                normalize])


pretrained_net = models.resnet18(pretrained=True)
# print(pretrained_net.fc)
pretrained_net.fc = torch.nn.Linear(512, 2)

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
lr = 0.01
optimizer = optim.SGD([{"params": feature_params},
                       {"params": pretrained_net.fc.parameters(), "lr": lr}],
                      lr=lr, weight_decay=0.001)


def train_fine_tuning(net, optimizer, batch_size=32, num_epoch=5):
    train_iter = DataLoader(ImageFolder(os.path.join(DATA_DIR, "train"),
                            transform=train_augs), batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(DATA_DIR, "test"),
                           transform=test_augs), batch_size=batch_size)
    loss = torch.nn.CrossEntropyLoss()
    train(train_iter, test_iter, net, loss, optimizer, device, num_epoch)


train_fine_tuning(pretrained_net, optimizer)

