from PIL import Image
import numpy as np
import math
import torch
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
from Code import ROOT
from Code.Utils.bboxes import show_bboxes
# print(torch.__version__)

img = Image.open(os.path.join(ROOT, "Datasets", "catdog.jpg"))
w, h = img.size

# 728,561
# print("w: %d, h:%d" % (w, h))


def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    device = feature_map.device
    pairs = []
    for r in ratios:
        pairs.append((sizes[0], math.sqrt(r)))
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])
    pairs = np.array(pairs)

    ss1 = pairs[:, 0] * pairs[:, 1]  # size * sqrt(ration)
    ss2 = pairs[:, 0] / pairs[:, 1]  # size / sqrt(ration)

    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2

    h, w = feature_map.shape[-2:]
    shifts_x = np.arange(0, w) / w
    shifts_y = np.arange(0, h) / h

    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)

    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4).to(device)


if __name__ == '__main__':

    X = torch.Tensor(1, 3, h, w)
    Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    # print(Y.shape) # torch.Size([1, 2042040, 4])

    boxes = Y.reshape((h, w, 5, 4))
    # print(boxes[250][250][0][:]) # tensor([-0.0316,  0.0706,  0.7184,  0.8206])

    fig = plt.imshow(img)
    bbox_scale = torch.tensor([w, h, w, h], dtype=torch.float32)
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ["s=0.75, r=1", "s=0.75, r=2", "s=0.55, r=0.5", "s=0.5, r=1", "s=0.25, r=1"])
    plt.show()