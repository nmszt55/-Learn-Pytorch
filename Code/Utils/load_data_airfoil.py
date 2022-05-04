import numpy as np
import torch
import os


CUR = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(os.path.dirname(os.path.dirname(CUR)), "Datasets", "airfoil_self_noise.dat")


def get_data():
    data = np.genfromtxt(TXT, delimiter="\t")
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
            torch.tensor(data[:1500, -1], dtype=torch.float32)
