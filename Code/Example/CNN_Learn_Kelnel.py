from Code.CNN.Conv2D import *

x = torch.ones((6, 8))
x[2:4, 4:6] = 0
y = corr2d(x, torch.tensor([[1, -1]]))
c = Conv2d(kernel_size=(1, 2))
# 学习次数
step = 200
# 学习率
lr = 0.005

for t in range(step):
    y_hat = c(x)
    loss = ((y_hat - y)**2).sum()
    loss.backward()
    c.weight.data -= lr * c.weight.grad
    c.bias.data -= lr * c.bias.grad

    c.weight.grad.zero_()
    c.bias.grad.zero_()
    if (t + 1) % 5 == 0:
        print("step %d, loss: %.3f" % (t+1, loss.item()))


print(c.weight)