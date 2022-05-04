from Code.Utils.common import train_2d, show_trace_2d, f_2d


def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2


eta, gamma = 0.6, 0.5
show_trace_2d(f_2d, train_2d(momentum_2d))
