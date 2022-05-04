import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


# gradian descend
def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        # x -= eta * f'(x)
        x -= eta * 2 * x
        results.append(x)
    print("epoch 10:", x)
    return results


def show(array):
    n = max(abs(min(array)), abs(max(array)), 10)
    f_line = np.arange(-n, n, 0.1)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(f_line, [x**2 for x in f_line])
    ax.plot(array, [x**2 for x in array], "-o")
    ax.set(xlabel="x", ylabel="y", title="$f(x)=x^2$")
    plt.show()


if __name__ == '__main__':
    show(gd(1.1))
