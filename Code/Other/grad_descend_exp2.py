import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


def train_2d(trainer):
    x1, x2 = -5, -2
    s1, s2 = 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print("epoch %d, x1 %.2f x2 %.2f" % (i+1, x1, x2))
    return results


def show_trace_2d(f, results):
    plt.plot(*zip(*results), "-o", color="#ff7f0e")
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors="#1f77b4")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


eta = 0.1


# 定义目标函数
def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)



show_trace_2d(f_2d, train_2d(gd_2d))
