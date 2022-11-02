import os
import torch
import torchtext.vocab as vocab
from Code import DATADIR


# print(vocab.pretrained_aliases.keys())
# print([key for key in vocab.pretrained_aliases.keys()
#        if "glove" in key])
cache_dir = os.path.join(DATADIR, "glove_6b_50d")
glove = vocab.GloVe(name="6B", dim=50, cache=cache_dir)
# print("一共有%d个词向量" % len(glove.stoi))


# print((glove.stoi["computer"], glove.itos[3355]))
def knn(w, x, k):
    cos = torch.matmul(w, x.view((-1,))) / ((torch.sum(w * w, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    _, topk = torch.topk(cos, k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]


def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.vectors, embed.vectors[embed.stoi[query_token]], k+1)
    for i, c in zip(topk[1:], cos[1:]):
        print("Cosine sim=%.3f: %s" % (c, embed.itos[i]))

get_similar_tokens('chip', 3, glove)
