import matplotlib

matplotlib.use("TkAgg")

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# 第三个值为实数表步长，第三个数为复数表形状
x, y = np.mgrid[-1:1:31j, -1:1:31j]
z = x**2 - y**2

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_wireframe(x, y, z, rstride=2, cstride=2)
ax.plot([0], [0], [0], "rx")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()