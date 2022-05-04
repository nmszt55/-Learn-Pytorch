import torch
from torch import nn, optim

from Code.Utils.common import train_pytorch, plt
from Code.Utils.load_data_airfoil import get_data

features, labels = get_data()
train_pytorch(optim.SGD, {"lr": 0.004, "momentum": 0.9}, features, labels)
plt.show()