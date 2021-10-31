"""使用torch的api实现歌词预测"""
import time
import math
import numpy as np
import sys
import torch

from torch import nn, optim
import torch.nn.functional as F

from Code.Utils.load_data_jay_lyrics import load_data_jay_lyrics
from Code.Utils.one_hot import to_onehot
from Code.Utils.predict_rnn import predict_rnn_torch, train_and_predict_rnn_torch
from Code.CNN.RNN import RNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


index_list, char2idx, index2char, vocab_size = load_data_jay_lyrics()


# -- 定义模型 --
num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

num_steps = 35
batch_size = 2
state = None
X = torch.rand(num_steps, batch_size, vocab_size)


if __name__ == '__main__':
    rnn = RNNModel(rnn_layer, vocab_size)
    train_and_predict_rnn_torch(rnn, num_hiddens, vocab_size, device, index_list, index2char, char2idx, 250,
                                num_steps, 0.001, 0.01, batch_size, 50, 50, ["分开", "不分开"])
