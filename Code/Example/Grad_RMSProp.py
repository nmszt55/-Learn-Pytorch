import math
import torch
import numpy as np

from Code.Utils.common import show_trace_2d, train_2d


def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1-gamma) * g1 ** 2
    s2 = gamma * s2 + (1-gamma) * g2 ** 2
    x1 -= eta / (math.sqrt(s1 + eps)) * g1
    x2 -= eta / (math.sqrt(s2 + eps)) * g2
    return x1, x2, s1, s2


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
show_trace_2d(f_2d, train_2d(rmsprop_2d))
