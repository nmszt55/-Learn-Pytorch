import numpy as np
import time
import torch
from torch import nn, optim
import sys
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


CUR = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(os.path.dirname(os.path.dirname(CUR)), "Datasets", "airfoil_self_noise.dat")


def get_data():
    data = np.genfromtxt(TXT, delimiter="\t")
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
            torch.tensor(data[:1500, -1], dtype=torch.float32)


features, labels = get_data()
# features.shape  --  [1500, 5]


def sgd(params, status, hyperparams):
    for p in params:
        p.data -= hyperparams["lr"] * p.grad.data


def linereg(X, w, b):
    return torch.mm(X, w) + b


def squared_loss(y, label):
    return 0.5 * (y - label.view(y.size())) ** 2


def train(optimizer_fn, status, hyperparams, features, labels, batchsize=10, num_epoch=2):
    # 初始化模型
    net, loss = linereg, squared_loss
    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1,dtype=torch.float32), requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batchsize, shuffle=True)
    for _ in range(num_epoch):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()

            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            optimizer_fn([w, b], status, hyperparams)
            if (batch_i + 1) * batchsize % 100 == 0:
                # 每100个样本记录当前误差
                ls.append(eval_loss())
        # 打印结果和作图
        print("Loss: %f, %f sec per epoch." % (ls[-1], time.time()-start))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, num_epoch, len(ls)), ls)
    ax.set(xlabel="epochs", ylabel="loss")
    ax.grid()


if __name__ == '__main__':
    features, labels = get_data()
    train(sgd, None, {"lr": 0.005}, features, labels, 10, 10)
    plt.show()

