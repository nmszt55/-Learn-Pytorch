import torch
from torch import nn, optim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

from Code.Other.sgd2 import linereg, squared_loss
from Code.Utils.load_data_airfoil import get_data

matplotlib.use("TkAgg")


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


def train_pytorch(optimizer_fn, optimizer_hyperparams, features, labels, batch_size=10, num_epoch=2):
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True
    )

    for _ in range(num_epoch):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X).view(-1), y) / 2
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i+1) * batch_size % 100 == 0:
                ls.append(eval_loss())
        print("loss: %f, %f sec per epoch.." % (ls[-1], time.time()-start))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, num_epoch, len(ls)), ls)
    ax.set(xlabel="epochs", ylabel="loss")
    ax.grid()


def show_trace_2d(f, results):
    plt.plot(*zip(*results), "-o", color="red")
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-5.5, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), color="blue")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def train_2d(trainer):
    x1, x2 = -5, -2
    s1, s2 = 0, 0
    results = [(x1,x2)]
    for i in range(20):
        x1,x2,s1,s2 = trainer(x1,x2,s1,s2)
        results.append((x1,x2))
    print("epoch %d, x1=%f x2=%f" % (i, x1, x2))
    return results


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


if __name__ == '__main__':
    features, labels = get_data()
    train_pytorch(optim.SGD, {"lr": 0.05}, features, labels, 10, 2)
    plt.show()