import numpy.random
import torch

from Code.Utils.benchmark import BenchMark

# GPU计算热身
device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)


with BenchMark("numpy"):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)


with BenchMark("torch"):
    for _ in range(10):
        a = torch.randn((1000, 1000), device=device)
        b = torch.mm(a, a)
