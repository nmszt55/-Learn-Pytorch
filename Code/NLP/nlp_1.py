"""
本节是对NLP前两节内容的实践。我们以跳字模型和近似训练中的负采样为例，介绍在语料库中训练词嵌入模型。
"""
import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data

print(torch.__version__)
from Code import DATADIR

data_dir = os.path.join(DATADIR, "ptb")
assert os.path.isdir(data_dir)

with open(os.path.join(data_dir, "ptb.train.txt")) as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]

print("# Len of sequence:", len(raw_dataset))
# for st in raw_dataset[:3]:
#     print("# tokens: ", len(st), st[:5])

counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(counter)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in raw_dataset]

num_tokens = sum([len(st) for st in dataset])
print("# tokens number:", num_tokens)


def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * num_tokens)


subsampled_datasets = [[tk for tk in st if not discard(tk)] for st in dataset]
print("# subsampled token: %d" % sum([len(st) for st in subsampled_datasets]))


def compare_counts(token):
    return "# %s: before=%d, after=%d" % (token, sum([st.count(token_to_idx[token]) for st in dataset]),
                                          sum([st.count(token_to_idx[token]) for st in subsampled_datasets]))

# print(compare_counts('the'))
print(compare_counts("join"))


def get_center_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size), min(len(st), center_i+1+window_size)))
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


# tiny_dataset = [list(range(7)), list(range(7, 10))]
# print("dataset", tiny_dataset)
# for center, context in zip(*get_center_and_contexts(tiny_dataset, 2)):
#     print("center", center, "has contexts", context)
all_centers, all_contexts = get_center_and_contexts(subsampled_datasets, 5)


def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重，随机生成k个词的索引作为噪声词
                # 为了高效计算，可以将k设置的较大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5)
                )
            neg, i = neg_candidates[i], i + 1
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return self.centers[index], self.contexts[index], self.negatives[index]

    def __len__(self):
        return len(self.centers)


def batchify(data):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list,
       list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives), \
            torch.tensor(masks), torch.tensor(labels)


batch_size = 512
num_workers = 0 if sys.platform.startswith("win32") else 4
dataset = MyDataset(all_centers, all_contexts, all_negatives)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify, num_workers=num_workers)
# for batch in data_iter:
#     for name, data in zip(["centers", "context_negatives", "masks", "labels"], batch):
#         print(name, "shape:", data.shape)
#     break


# embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
# print(embed.weight)
# x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
# print(embed(x))

# X = torch.ones((2, 1, 4))
# Y = torch.ones((2, 4, 6))
# print(torch.bmm(X, Y).shape)


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0,2,1))

    return pred


# 二元交叉熵损失函数
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        input, target, mask = input.float(), target.float(), mask.float()
        res = F.binary_cross_entropy_with_logits(input, target, reduction="none", weight=mask)
        return res.mean(dim=1)


loss = SigmoidBinaryCrossEntropyLoss()
# pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# # 标签变量label中的1和0分别代表背景词和噪声词
# label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
# mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
# print(loss(pred, label, mask) * mask.shape[1] / mask.float().sum(dim=1))


embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)
)


def train(net, lr, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("train on %s" % device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.view(label.shape), label, mask) * mask.shape[1]/mask.float().sum(dim=1)).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            l_sum += l.cpu().item()
            n += 1
        print("epoch %d, loss %.2f, time %.1fs" % (epoch + 1, l_sum / n, time.time() - start))


def get_similar_tokens(query_token, k, embed):
    w = embed.weight.data
    x = w[token_to_idx[query_token]]
    # 这个1e-9是为了增加数值稳定性
    cos = torch.matmul(w, x) / (torch.sum(w * w, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    # 除去原词
    for i in topk[1:]:
        print("cosine: sim=%.3f: %s" % (cos[i], idx_to_token[i]))


if __name__ == '__main__':
    train(net, 0.01, 10)
    get_similar_tokens("chip", 3, net[0])