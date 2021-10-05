import torch


def one_hot(x, n_class, dtype=torch.float32):
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    # scatter_(dim, index, src): 将src写到指定的dim中,其索引由第二个参数指定。
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


if __name__ == '__main__':
    from pprint import pprint
    x = torch.arange(10).view(2, 5)
    print(x)
    pprint(to_onehot(x, 10))