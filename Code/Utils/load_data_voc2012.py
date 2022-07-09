import sys

import urllib3
import os
import tqdm
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image
from Code.Utils.image_augmentation import show_images, plt

fileurl = "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def download(dst_file):
    conn_pool = urllib3.PoolManager()
    resp = conn_pool.request("GET", fileurl, preload_content=False)
    check_sum = 0
    buff_size = 1024
    with open(dst_file, "wb") as f:
        for i, buf in enumerate(resp.stream(buff_size), start=1):
            f.write(buf)
            check_sum += buff_size
            if i % 1000 == 0:
                print("\r")
                print("download %.2f MB" % (check_sum / i / 1024 / 1024), end="")
    print("download success!!")


def read_voc_images(root, is_train, max_num=None):
    root = os.path.join(root, "VOCdevkit", "VOC2012")
    txt_fname = os.path.join(root, "ImageSets", "Segmentation", "train.txt" if is_train else "val.txt")
    with open(txt_fname) as f:
        images = f.read().split()
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm.tqdm(enumerate(images)):
        features[i] = Image.open(os.path.join(root, "JPEGImages", fname+".jpg")).convert("RGB")
        labels[i] = Image.open(os.path.join(root, "SegmentationClass", fname+".png")).convert("RGB")
    return features, labels


colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i


def voc_label_indices(colormap, colormap2label):
    colormap = np.array(colormap.convert("RGB")).astype("int32")
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


def voc_rand_crop(feature, label, h, w):
    """random crop feature(PIL Image) and label(PIL Image)"""
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(feature, output_size=(h, w))
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        # https://blog.csdn.net/qq_41076797/article/details/111005890
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        ])
        self.crop_size = crop_size
        features, labels = read_voc_images(root=voc_dir, is_train=is_train, max_num=max_num)
        self.features = self.filter(features)
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label
        print("read " + str(len(features)) + "vaild examples")

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and
            img.size[0] >= self.crop_size[1]
        )]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return self.tsf(feature), voc_label_indices(label, self.colormap2label)

    def __len__(self):
        return len(self.features)




if __name__ == '__main__':
    n = 5
    root = "/home/zhaozijian/Downloads/VOCtrainval_11-May-2012"
    # features, labels = read_voc_images(root, True, 100)
#    show_images(features[0:n]+labels[0:n], 2, n)
#    plt.show()
#     y = voc_label_indices(labels[0], colormap2label)
    # print(y[105: 115, 130: 140], VOC_CLASSES[1])
    # imgs = []
    # for _ in range(n):
    #     imgs += voc_rand_crop(features[0], labels[0], 200, 300)
    # show_images(imgs[::2] + imgs[1::2], 2, 5)
    # plt.show()

    crop_size = (320, 480)
    max_num = 100
    voc_train = VOCSegDataset(True, crop_size, root, colormap2label, max_num)
    voc_test = VOCSegDataset(False, crop_size, root, colormap2label, max_num)
    batch_size = 64
    num_worker = 0 if sys.platform.startswith("win32") else 4
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                             drop_last=True, num_workers=num_worker)
    test_iter = torch.utils.data.DataLoader(voc_test, batch_size, drop_last=True, num_workers=num_worker)

    for X, y in train_iter:
        print(X.shape, X.dtype)
        print(y.shape, y.dtype)
        break