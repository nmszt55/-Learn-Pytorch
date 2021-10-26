import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, optim
from Code.Utils.load_data_jay_lyrics import load_data_jay_lyrics
from Code.Utils.one_hot import to_onehot
from Code.Utils.predict_rnn import predict_rnn

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
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def test1():
    """不训练模型直接预测"""
    params = get_params()
    state = init_rnn_state(1, num_hidden, device)
    res = predict_rnn("分开", 10, rnn, params, init_rnn_state, num_hidden, vocab_size, device, idx2char, char2idx)
    print(res)


def test2():
    """训练 + 预测"""
    from Code.Utils.predict_rnn import train_and_predict_rnn
    num_epoch, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ["分开", "天晴"]
    train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hidden, vocab_size, device, index_list,
                          idx2char, char2idx, True, num_epoch, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes)


if __name__ == '__main__':
    test2()