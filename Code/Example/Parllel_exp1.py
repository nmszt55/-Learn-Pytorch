import torch
import time
from Code.Utils.benchmark import BenchMark

assert torch.cuda.device_count() >= 2


def run(x):
    for _ in range(20000):
        y = torch.mm(x, x)


x_gpu1 = torch.rand(size=(100, 100), device="cuda:0")
x_gpu2 = torch.rand(size=(100, 100), device="cuda:1")

with BenchMark("GPU1"):
    run(x_gpu1)
    torch.cuda.synchronize()

with BenchMark("GPU2"):
    run(x_gpu2)
    torch.cuda.synchronize()
