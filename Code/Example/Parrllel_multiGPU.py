import torch

net = torch.nn.Linear(10, 1).cuda()
print(net)

net = torch.nn.DataParallel(net)
print(net)

