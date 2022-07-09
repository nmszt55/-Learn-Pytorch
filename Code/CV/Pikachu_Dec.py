import matplotlib.pyplot as plt

from Code.Utils.bbox_dataloader import *
from Code.Utils.bboxes import show_bboxes
from Code.Utils.image_augmentation import show_images


batch_size, edge_size = 32, 256
train_iter, _ = load_data_pikachu(batch_size, edge_size)
batch = iter(train_iter).next()
# print(batch["image"].shape, batch["label"].shape)

imgs = batch["image"][0:10].permute(0, 2, 3, 1)
bboxes = batch["label"][0:10, 0, 1:]
axes = show_images(imgs, 2, 5).flatten()
for ax, bb in zip(axes, bboxes):
    show_bboxes(ax, [bb * edge_size], colors="w")
plt.show()
