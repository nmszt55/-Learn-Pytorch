import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from Code.Utils.common import show_trace_2d


matplotlib.use("TkAgg")

eta = 0.1


def sgd_2d(x1, x2, s1, s2):
    return x1 - eta * (2 * x1 + np.random.normal(0.1)),\
           x2 - eta * (4 * x2 + np.random.normal(0.1)),\
           0, 0


def train_2d(trainer):
    x1, x2 = 5, -2
    s1, s2 = 0, 0
    results = [(x1,x2)]
    for i in range(20):
        x1,x2,s1,s2 = trainer(x1,x2,s1,s2)
        results.append((x1,x2))
    print("epoch %d, x1=%.2f x2=%.2f" % (i, x1, x2))
    return results


def show_trace_2d(f, results):
    plt.plot(*zip(*results), "-o", color="red")
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-5.5, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), color="blue")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


if __name__ == '__main__':
    show_trace_2d(f_2d, train_2d(sgd_2d))