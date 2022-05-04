"""
本代码展示使用梯度下降的问题
"""
import sys
import torch

from Code.Utils.common import show_trace_2d, train_2d


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return x1 - (eta * 0.2 * x1), x2 - (eta * 4 * x2), 0, 0


if __name__ == '__main__':
    eta = 0.4
    show_trace_2d(f_2d, train_2d(gd_2d))