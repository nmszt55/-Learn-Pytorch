import os
from torch.utils.data import DataLoader
import torchvision
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CIFAR_PATH = os.path.join(ROOT, "Datasets", "cifar-10")


num_worker = 0 if sys.platform.startswith("win32") else 4

def load_cifar10(is_train, augs, batchsize, root=CIFAR_PATH):
    dataset = torchvision.datasets.CIFAR10(root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batchsize, shuffle=is_train, num_workers=num_worker)