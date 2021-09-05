from Code.CNN.corr2d import *


class Conv2d(torch.nn.Module):
    def __init__(self, kernel_size:(int, int)):
        super(Conv2d, self).__init__()
        # definite kernel
        self.weight = torch.nn.Parameter(torch.randn(kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

