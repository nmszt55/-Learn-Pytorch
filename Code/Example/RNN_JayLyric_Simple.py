"""使用torch的api实现歌词预测"""
import time
import math
import numpy as np
import sys
import torch

from torch import nn, optim
import torch.nn.functional as F

from Code.Utils.load_data_jay_lyrics import load_data_jay_lyrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


index_list, char2idx, index2char, vocab_size = load_data_jay_lyrics()


# -- 定义模型 --
num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)