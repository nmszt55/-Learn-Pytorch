import numpy
import numpy as np
import sys
import torch
import torch.nn.functional as F

from torch import nn, optim
from Code.RNN.LSTM import lstm
from Code.Utils.load_data_jay_lyrics import load_data_jay_lyrics
from Code.Utils.predict_rnn import train_and_predict_rnn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("will use %s" % device)

corpus_index, char2idx, idx2char, vocab_size = load_data_jay_lyrics()

# 定义初始化参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size


def get_params():
    def one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), dtype=torch.float32, device=device)
        return torch.nn.Parameter(ts, requires_grad=True)

    def three():
        return (one((num_inputs, num_hiddens)),
                one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, dtype=torch.float32, device=device), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])


def init_lstm_status(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


num_epoch = 1200
num_steps = 35
batch_size = 64
lr = 1e2
clipping_theta = 1e-2

pred_period = 40
pred_len = 50
prefixes = ["分开", "不分开"]

train_and_predict_rnn(lstm, get_params, init_lstm_status, num_hiddens, vocab_size, device, corpus_index,
                      idx2char, char2idx, False, num_epoch, num_steps, lr, clipping_theta, batch_size, pred_period,
                      pred_len, prefixes)

