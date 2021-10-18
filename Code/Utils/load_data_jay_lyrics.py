"""
该代码为加载周杰伦歌词数据的方法包，用于RNN训练
可以查看Code/Example/RNN_JayLyric.py,此模块实现了RNN的训练
"""
import zipfile
import os
import random
import torch

Cur = os.path.dirname(os.path.abspath(__file__))
Root = os.path.dirname(os.path.dirname(Cur))


# 1. 数据集加载
def get_lyric_string(filepath=os.path.join(Root, "Datasets", "jaychou_lyrics.txt.zip")):
    if not os.path.isfile(filepath):
        raise FileNotFoundError("Cant find datasets of jaychou_lyrics")
    with zipfile.ZipFile(filepath) as zip:
        with zip.open("jaychou_lyrics.txt") as f:
            corpus_chars = f.read().decode("utf-8")
            return corpus_chars


def get_string_dict(string_list):
    dic = {char: i for i, char in enumerate(string_list)}
    return dic


def string2idx(string, dic):
    return [dic[x] for x in string]


def load_data_jay_lyrics():
    """加载歌词数据集"""
    strings = get_lyric_string()[:10000]
    string_chars = strings.replace('\n', ' ').replace('\r', ' ')
    string_chars = string_chars[0:10000]
    idx2char = list(set(string_chars))
    char2idx = get_string_dict(idx2char)
    vocab_size = len(char2idx)
    indics = [char2idx[x] for x in string_chars]
    return indics, char2idx, idx2char, vocab_size


def data_iter_random(index_list, batch_size, num_steps, device=None):
    """
    实现数据集随机采样
    index_list: 字符串转换成index后的列表,函数会从该list中抽取连续的字符片段作为训练的输入
    batch_size: 批量大小
    num_steps: 时间步大小
    device: 数据设备
    """
    num_example = (len(index_list) - 1) // num_steps
    epoch_size = num_example // batch_size
    example_indices = list(range(num_example))
    random.shuffle(example_indices)

    # 返回从pos开始的，长度为num_steps的序列
    def _data(pos):
        return index_list[pos: pos + num_steps]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i+batch_size]
        X = [_data(j*num_steps) for j in batch_indices]
        Y = [_data(j*num_steps+1) for j in batch_indices]
        yield (torch.tensor(X, dtype=torch.float32, device=device),
               torch.tensor(Y, dtype=torch.float32, device=device))


def data_iter_consecutive(index_list, batch_size, num_steps, device=None):
    """实现相邻采样"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index_list = torch.tensor(index_list, dtype=torch.float32, device=device)
    data_len = len(index_list)
    batch_len = data_len // batch_size
    indices = index_list[0: batch_size * batch_len].view((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i:i+num_steps]
        Y = indices[:, i+1: i+num_steps+1]
        yield X, Y



if __name__ == '__main__':
    x = list(range(30))
    for x, y in data_iter_consecutive(x, 2, 6):
        print(x)
        print(y)
        print("---")