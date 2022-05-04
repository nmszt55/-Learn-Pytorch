import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits import mplot3d


matplotlib.use("TkAgg")

def f(x):
    return x * np.cos(np.pi * x)


fig, ax = plt.subplots()
x = np.arange(-1.0, 2.0, 0.1)
ax.annotate("local minimum", xy=(-0.3, -0.25), xytext=(-0.77, -1.0), arrowprops=dict(arrowstyle="->"))
ax.annotate("global minimum", xy=(1.1, -0.95), xytext=(0.6, 0.8), arrowprops=dict(arrowstyle="->"))
ax.plot(x, f(x), linewidth=2.0)

plt.show()