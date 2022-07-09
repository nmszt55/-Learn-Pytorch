import torch
import torchvision

X = torch.arange(16, dtype=torch.float32).view(1,1,4,4)
# print(X)

rois = torch.tensor([[0,0,0,20,20], [0,0,10,30,30]], dtype=torch.float)
print(torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1))