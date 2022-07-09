import torch
import os
from PIL import Image

from Code import ROOT
from Code.Utils.bboxes import show_bboxes, plt, MultiBoxDetection


anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
cls_probs = torch.tensor([[0., 0., 0., 0.],  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
img = Image.open(os.path.join(ROOT, "Datasets", "catdog.jpg"))
bbox_scale = torch.tensor(img.size * 2, dtype=torch.float32)  # (728, 561)
# fig = plt.imshow(img)
# show_bboxes(fig.axes, anchors*bbox_scale, ["dog=0.9", "dog=0.8", "dog=0.7", "cat=0.9"])
# plt.show()

output = MultiBoxDetection(
    cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(0),
    anchors.unsqueeze(0), nms_threshold=0.5
)
# print(output)

fig = plt.imshow(img)
for i in output[0].detach().cpu().numpy():
    if i[0] == -1:
        continue
    label = ("dog=", "cat=")[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
plt.show()