import torch

from Code.CNN.corr2d import corr2d


def corr2d_multi_pipe(X, K):
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res


def corr2d_multi_outpipe(X, K):
    return torch.stack([corr2d_multi_pipe(X, K[i]) for i in range(K.shape[0])])


if __name__ == '__main__':
    X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

    # print(corr2d_multi_pipe(X, K))
    K = torch.stack([K, K + 1, K + 2])
    print(K.shape)  # torch.Size([3, 2, 2, 2])
    res = corr2d_multi_outpipe(X, K)
    print(res)
    print(res.shape)