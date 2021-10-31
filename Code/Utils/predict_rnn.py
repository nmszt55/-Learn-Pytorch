"""
此模块包含了RNN的训练方法
"""
import torch
import time
import math
import numpy as np

from Code.Utils.one_hot import to_onehot
from Code.Utils.sgd import sgd
from Code.Utils.load_data_jay_lyrics import data_iter_random, data_iter_consecutive


def init_rnn_state(batch_size, num_hidden, device):
    """初始化最开始的隐藏状态，值为0"""
    return (torch.zeros(batch_size, num_hidden, device=device), )


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens,
                vocab_size, device, idx_to_char, char_to_idx):
    """
    prefix: 前面已存在的字符
    num_chars: rnn需要前向预测的字符数。
    rnn: 循环模型
    params: 权重列表，为(权重，隐藏状态权重，偏差，输出层权重，输出层偏差)
    init_rnn_state: 初始化隐藏状态的方法，为function类型
    num_hiddens:
    vocab_size: 将所有字符提取出来，汇总具有的字符的总数
    device: 设备类型，由torch.device初始化
    idx_to_char: {index:字符}字典，将index转换为字符。
    char_to_idx: {字符:index}字典，作用是将字符串转换成int类型的字典。
    """
    state = init_rnn_state(1, num_hiddens, device)
    # 添加第一个字符
    output = [char_to_idx[prefix[0]]]
    for t in range(len(prefix) + num_chars - 1):
        # 将上一时间步的输出作为这一时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和隐藏状态。
        Y, state = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            # 本训练批次设置为1，所以只需获取Y[0]即可
            output.append(int(Y[0].argmax(dim=1).item()))
    return "".join(idx_to_char[i] for i in output)


def grad_clipping(params, theta, device):
    """梯度裁剪"""
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def rnn(input, state, params):
    """实现rnn的循环计算"""
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for x in input:
        # tanh激活函数
        H = torch.tanh(torch.matmul(x, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size,
                          device, corpus_indices, idx_to_char, char_to_idx, is_random_iter,
                          num_epoch, num_step, lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        # 随机采样
        data_iter_fn = data_iter_random
    else:
        # 连续采样
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epoch):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, device)
        # 总损失，计数，计时
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_step, device)
        for x, y in data_iter:
            if is_random_iter:
                # 随机采样的批次数据连续，所以每次训练完成之后重设初始状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                for s in state:
                    # 分离计算图，减轻训练复杂度
                    s.detach_()
            inputs = to_onehot(x, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的list
            output, state = rnn(inputs, state, params)
            # 将list的所有tensor按照行为单位进行拼接
            output = torch.cat(output, dim=0)
            # 计算完成后，Y是一个长度为batch*num_steps的向量
            Y = torch.transpose(y, 0, 1).contiguous().view(-1)
            # 交叉熵损失计算误差
            l = loss(output, Y.long())
            # 梯度清零
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)
            sgd(params, lr, 1)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print("epoch %d, perplexity %f, time %.2f sec" % (epoch, math.exp(l_sum / n), time.time()-start))
            for prefix in prefixes:
                print("-", predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size,
                                       device, idx_to_char, char_to_idx))


def predict_rnn_torch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    """基于torch的Module进行预测"""
    state = None
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        Y, state = model(X, state)
        if t < len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return "".join([idx_to_char[x] for x in output])


def train_and_predict_rnn_torch(model, num_hidden, vocab_size, device, corpus_indices,
                                idx_to_char, char_to_idx, num_epoch, num_steps, lr,
                                clipping_theta, batch_size, pred_period, pred_len, prefixes):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epoch):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if state is not None:
                # 使用detach使隐藏状态从计算图分离,这是为了
                # 使模型参数的梯度计算只依赖一次迭代的小批量序列，减小开销。
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            output, state = model(X, state)

            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())
            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            # 执行梯度下降
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        try:
            # 评估困惑度
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float("inf")
        if (epoch + 1) % pred_period == 0:
            print("epoch %d, the perplexity is %f, time %.2f" % (epoch, perplexity, time.time()-start))
            for prefix in prefixes:
                print(" - " + predict_rnn_torch(prefix, pred_len, model, vocab_size, device, idx_to_char,
                                        char_to_idx))
