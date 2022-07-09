from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import sys
from Code import ROOT
from Code.CV.anchor_box import MultiBoxPrior
from Code.Utils.bboxes import show_bboxes

img = Image.open(os.path.join(ROOT, "Datasets", "catdog.jpg"))
w, h = img.size


def display_anchors(fmap_w, fmap_h, s):
    # 前两维的取值不影响输出结果
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)

    # 平移所有锚框使之均匀分布与图片中
    offset_w, offset_h = 1.0/fmap_w, 1.0/fmap_h
    anchors = MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) + \
        torch.tensor([offset_w / 2, offset_h / 2, offset_w / 2, offset_h / 2])
    # print(anchors.shape)
    bbox_scale = torch.tensor([w, h, w, h], dtype=torch.float32)
    show_bboxes(plt.imshow(img).axes, anchors[0] * bbox_scale)


if __name__ == '__main__':
    display_anchors(1, 1, [0.8])
    plt.show()