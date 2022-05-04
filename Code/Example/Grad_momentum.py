import torch

from Code.Utils.load_data_airfoil import get_data

features, labels = get_data()


def init_momentum_status():
    v_w = torch.zeros(features.shape[1], 1)
    v_b = torch.zeros(1, dtype=torch.float32)
    return v_w, v_b


def sgd_momentum(params, status, hyperparams):
    for p, v in zip(params, status):
        v.data = hyperparams["momentum"] * v.data + hyperparams["lr"] * p.grad.data
        p.data -= v.data


if __name__ == '__main__':
    from Code.Utils.common import train, plt
    train(sgd_momentum, init_momentum_status(), {"lr": 0.004, "momentum": 0.9}, features, labels)
    plt.show()