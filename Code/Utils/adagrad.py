from Code.Utils.load_data_airfoil import get_data
from Code.Utils.common import train, plt, train_pytorch

import torch


def init_ada_status(feature_shape=None):
    if not feature_shape:
        feature_shape = get_data()[0].shape
    s_w = torch.zeros(feature_shape[1], 1, dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return s_w, s_b


def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s.data += p.grad.data ** 2
        p.data -= hyperparams["lr"] * p.grad.data / torch.sqrt(s + eps)


# if __name__ == '__main__':
#     features, labels = get_data()
#     train(adagrad, init_ada_status(), {"lr": 0.1}, features, labels)
#     plt.show()

if __name__ == '__main__':
    features, labels = get_data()
    train_pytorch(torch.optim.Adagrad, {"lr": 0.1}, features, labels)
    plt.show()
