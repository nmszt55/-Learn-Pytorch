import torch

from Code.Utils.one_hot import to_onehot


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
    norm = torch.tensor([0, 0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)