import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from Code import ROOT
from Code.Utils.bboxes import show_bboxes, MultiBoxTarget

matplotlib.use("TkAgg")
img = Image.open(os.path.join(ROOT, "Datasets", "catdog.jpg"))
w, h = img.size

bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
# 真实边界框:[种类,左上x,左上y,右下x,右下y]
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                            [1, 0.55, 0.2, 0.9, 0.88]])
# 锚框
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

# fig = plt.imshow(img)
# show_boxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ["dog", "cat"], "k")
# show_boxes(fig.axes, anchors * bbox_scale, ["0", "1", "2", "3", "4"])
# plt.show()

labels = MultiBoxTarget(anchors.unsqueeze(dim=0),
                        ground_truth.unsqueeze(dim=0))
print(labels[2])
print(labels[1])