import sys
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, optim

from Code.Utils.load_data_jay_lyrics import load_data_jay_lyrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Will use %s" % device)

corpus_indeces, char2idx, idx2char, vocab_size = load_data_jay_lyrics()


num_input, num_hidden, num_output = vocab_size, 256, vocab_size


def get_params():
    def one(shape):
        tensor = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return nn.Parameter(tensor, requires_grad=True)

    def three():
        return one((num_input, num_hidden)), \
               one((num_hidden, num_hidden)), \
               torch.nn.Parameter(torch.zeros(num_hidden, device=device, dtype=torch.float32), requires_grad=True)
    # 更新门参数
    W_xz, W_hz, b_z = three()
    # 重置门参数
    W_xr, W_hr, b_r = three()
    # 候选隐藏状态参数
    W_xh, W_hh, b_h = three()

    # 输出层参数
    W_hq = one((num_hidden, num_output))
    b_q = torch.nn.Parameter(torch.zeros(num_output, device=device, dtype=torch.float32).data, requires_grad=True)

    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def gru(input, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for x in input:
        Z = torch.sigmoid(torch.matmul(x, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(x, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(x, W_xh) + torch.matmul(R*H, W_hh) + b_h)
        H = Z*H + (1-Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


num_epoch, num_step, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ["分开", "不分开"]
from Code.Utils.predict_rnn import train_and_predict_rnn
train_and_predict_rnn(gru, get_params, init_gru_state, num_hidden, vocab_size, device, corpus_indeces, idx2char,
                      char2idx, False, num_epoch, num_step, lr,
                      clipping_theta, batch_size, pred_period, pred_len, prefixes)
