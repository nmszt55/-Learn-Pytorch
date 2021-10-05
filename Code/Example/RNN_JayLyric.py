import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, optim
from Code.Utils.load_data_jay_lyrics import load_data_jay_lyrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

index_list, char2idx, idx2char, vocab_size = load_data_jay_lyrics()

num_inputs, num_hidden, num_output = vocab_size, 256, vocab_size
print("Will use", device)


def get_params():
    # 初始化一个均值为0，标准差为0.01的tensor
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    # 隐藏层参数
    W_xh = _one((num_inputs, num_hidden))
    W_hh = _one((num_hidden, num_hidden))
    b_h = torch.nn.Parameter(torch.zeros(num_hidden, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hidden, num_output))
    b_q = torch.nn.Parameter(torch.zeros(num_output, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def init_rnn_state(batch_size, num_hidden, device):
    return (torch.zeros(batch_size, num_hidden, device=device), )


def rnn(input, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for x in input:
        # tanh激活函数
        H = torch.tanh(torch.matmul(x, W_xh) + torch.matmul(H, W_hh) + b_h)
