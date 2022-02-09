import numpy
import numpy as np
import sys
import torch
import torch.nn.functional as F

from torch import nn, optim
from Code.Utils.load_data_jay_lyrics import load_data_jay_lyrics
from Code.RNN.RNN import RNNModel
from Code.Utils.predict_rnn import train_and_predict_rnn_torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("will use %s" % device)

corpus_index, char2idx, idx2char, vocab_size = load_data_jay_lyrics()

# 定义初始化参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

lr = 1e-2
epochs = 200
num_steps = 35
clipping_theta = 1e-2
batch_size, pred_period, pred_len = 32, 40, 50
prefixes = ["分开", "不分开"]

lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(lstm_layer, vocab_size)
train_and_predict_rnn_torch(model, num_hiddens, vocab_size, device, corpus_index, idx2char,
                            char2idx, epochs, num_steps, lr, clipping_theta, batch_size, pred_period,
                            pred_len, prefixes)