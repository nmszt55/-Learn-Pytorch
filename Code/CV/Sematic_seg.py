"""
语义分割示例，使用pascal VOC2012
"""

import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

