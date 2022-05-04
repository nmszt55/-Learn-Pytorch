import matplotlib
import numpy as np
import matplotlib.pyplot as plt


matplotlib.use("TkAgg")


x = np.arange(-2.0, 2.0, 0.1)
fig, ax = plt.subplots()

ax.plot(x, x**3)
ax.annotate("saddle point", xy=(0, -0.2), xytext=(-0.52, -5.0), arrowprops=dict(arrowstyle="->"))

plt.show()