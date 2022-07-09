import torch
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, Dataset
import torchvision

import os
device = "cuda" if torch.cuda.is_available() else "cpu"
matplotlib.use("TkAgg")


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
img = Image.open(

    os.path.join(ROOT, "Datasets", "cat1.jpg"))
# plt.imshow(img)
# plt.show()


def show_images(imgs, num_rows, num_cols, scale=2):
    fig_size = (num_rows * scale, num_cols * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i*num_cols+j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


if __name__ == '__main__':
    # apply(img, torchvision.transforms.RandomVerticalFlip())
    # plt.show()

    # shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
    # apply(img, shape_aug)
    # plt.show()

    # 这里的0.5表示亮度为原来的随机(1-0.5, 1+0.5)区间
    # apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
    # plt.show()

    # 变化颜色
    # apply(img, torchvision.transforms.ColorJitter(hue=0.5))
    # plt.show()

    # 变化对比度
    # apply(img, torchvision.transforms.ColorJitter(contrast=0.5))
    # plt.show()

    # 变化所有
    # apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    # plt.show()

    # 叠加增广方法
    augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
    ])
    apply(img, augs)
    plt.show()