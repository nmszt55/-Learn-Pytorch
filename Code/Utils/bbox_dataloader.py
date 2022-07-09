"""
pikachu dataloader
"""
import os
import json
import numpy as np
import torch
import torchvision

from PIL import Image
from Code import ROOT


class PikachuDetDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, part, image_size=(256, 256)):
        assert part in ["train", "val"]
        self.image_size = image_size
        self.image_dir = os.path.join(datadir, part, "images")

        with open(os.path.join(datadir, part, "label.json")) as f:
            self.label = json.load(f)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        image_path = str(index + 1) + ".png"

        cls = self.label[image_path]["class"]
        label = np.array([cls] + self.label[image_path]["loc"], dtype="float32")[None, :]
        PIL_img = Image.open(os.path.join(self.image_dir, image_path)).convert("RGB").resize(self.image_size)
        img = self.transform(PIL_img)
        sample = {
            "label": label,
            "image": img
        }

        return sample


def load_data_pikachu(batch_size, edge_size=256, data_dir=os.path.join(ROOT, "Datasets", "pikachu")):
    """edge_size: 输出图像的宽和高"""
    image_size = (edge_size, edge_size)
    train_dataset = PikachuDetDataset(data_dir, "train", image_size)
    val_dataset = PikachuDetDataset(data_dir, "val", image_size)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_iter, val_iter
