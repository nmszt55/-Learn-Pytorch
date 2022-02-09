"""使用torch的api实现歌词预测"""
import torch

from torch import nn

from Code.Utils.load_data_jay_lyrics import load_data_jay_lyrics
from Code.Utils.predict_rnn import train_and_predict_rnn_torch
from Code.RNN.RNN import RNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


index_list, char2idx, index2char, vocab_size = load_data_jay_lyrics()


# -- 定义模型 --
num_epoch = 250

num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

num_steps = 35
batch_size = 32
lr = 1e-3
clipping_theta = 1e2
state = None
X = torch.rand(num_steps, batch_size, vocab_size)

predict_period = 50
predict_len = 50

if __name__ == '__main__':
    print("using %s" % device)
    rnn = RNNModel(rnn_layer, vocab_size)
    train_and_predict_rnn_torch(rnn, num_hiddens, vocab_size, device, index_list, index2char, char2idx, num_epoch,
                                num_steps, lr, clipping_theta, batch_size, predict_period, predict_len, ["爱你"])
