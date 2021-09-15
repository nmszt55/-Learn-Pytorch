import os
import sys
import torchvision

from torch.utils.data import DataLoader
from torchvision.transforms import transforms


CUR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CUR))


def get_data_fashion_mnist(batch_size, resize=None):
    trans = []
    # 重新设置size
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    train_mnist = torchvision.datasets.FashionMNIST(
        root=os.path.join(PROJECT_ROOT, "Datasets", "Fashion-Mnist"),
        train=True,
        download=True,
        transform=transform,
    )
    test_mnist = torchvision.datasets.FashionMNIST(
        root=os.path.join(PROJECT_ROOT, "Datasets", "Fashion-Mnist"),
        train=False,
        download=True,
        transform=transform,
    )

    if sys.platform.startswith("win"):
        num_worker = 0
    else:
        num_worker = 4
    train_iter = DataLoader(train_mnist, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_iter = DataLoader(test_mnist, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    return train_iter, test_iter


if __name__ == '__main__':
    train, test = get_data_fashion_mnist(256)
    for data, label in train:
        print(data)
        print(label)
        break


