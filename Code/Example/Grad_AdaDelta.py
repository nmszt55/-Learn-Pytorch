import torch
from Code.Utils.common import get_data, train, plt, train_pytorch


features, labels = get_data()


def init_adadelta_state():
    s_w, s_b = torch.zeros(features.shape[1], 1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    delta_w, delta_b = torch.zeros(features.shape[1], 1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return (s_w, delta_w), (s_b, delta_b)


def adadelta(params, states, hyperparams):
    rho, eps = hyperparams["rho"], 1e-5
    for p, (s, delta) in zip(params, states):
        s.data = rho * s + (1 - rho) * (p.grad.data ** 2)
        g = p.grad.data * torch.sqrt(delta + eps / s + eps)
        p.data -= g
        delta.data = rho * delta + (1 - rho) * g * g


# train(adadelta, init_adadelta_state(), {"rho": 0.9}, features, labels)
# plt.show()

train_pytorch(torch.optim.Adadelta, {"rho": 0.9}, features, labels)
plt.show()