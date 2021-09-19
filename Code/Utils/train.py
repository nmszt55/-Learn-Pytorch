import time
import torch

from Code.Utils.evaluate_accuracy import evaluate_acc


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epoch):
    """
    训练模型，并在训练过程中评估准确率
    :param net: 网络模型or前向传播方法
    :param train_iter: 训练数据集（需可拆分为X和Y）
    :param test_iter: 测试数据集，同训练集
    :param batch_size: 批量训练采用的数据集长度
    :param optimizer: 优化器
    :param device:类型字符串，描述设备名称
    :param num_epoch:训练循环次数
    """
    net = net.to(device)
    print("training on %s" % device)
    # 交叉熵损失函数
    loss = torch.nn.CrossEntropyLoss()
    for each in range(num_epoch):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0, 0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            print(y.shape)
            print(y_hat.shape)
            l = loss(y_hat, y)
            # 由于导数会累积，每次计算前清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # 存储在不同设备的数据不能直接进行计算，需要将数据搬到cpu中进行计算
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        # 每完成一次训练，在测试集统计一次准确率
        test_acc = evaluate_acc(test_iter, net, device)
        print("epoch %d, loss:%.3f, train acc:%.3f, test_acc:%.3f, time:%.1f sec" % (
            each+1, train_loss_sum/batch_count, train_acc_sum/n, test_acc, time.time()-start
        ))
