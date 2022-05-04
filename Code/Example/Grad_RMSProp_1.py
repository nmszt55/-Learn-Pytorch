import torch
from Code.Utils.common import get_data, train, plt, train_pytorch

features, labels = get_data()


def init_rmsprop_state():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return s_w, s_b


def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams["gamma"], 1e-6
    for p, s in zip(params, states):
        s.data = gamma * s.data + (1-gamma) * p.grad.data ** 2
        p.data -= hyperparams["lr"] * p.grad.data / torch.sqrt(s + eps)


if __name__ == '__main__':
    # train(rmsprop, init_rmsprop_state(), {"lr": 0.01, "gamma": 0.9}, features, labels)
    # plt.show()
    train_pytorch(torch.optim.RMSprop, {"lr": 0.01, "alpha": 0.9}, features, labels)
    plt.show()