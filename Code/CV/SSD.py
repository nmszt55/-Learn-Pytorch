import torch

from torch import nn
import torchvision
import os
from torch.functional import F
from Code import ROOT
from Code.Utils.bboxes import show_bboxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)


# Y1 = forward(torch.zeros(2, 8, 20, 20), cls_predictor(8, 5, 10))
# Y2 = forward(torch.zeros(2, 16, 10, 10), cls_predictor(16, 3, 10))
# print(Y1.shape, Y2.shape)


def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(pred) for pred in preds], dim=1)


# print(concat_preds([Y1, Y2]).shape)
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


# features = forward(torch.zeros((2,3,20,20)), down_sample_blk(3, 10))
# print(features.shape)


def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters)-1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


# features = forward(torch.zeros((2,3,256,256)), base_net())
# print(features.shape)

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


from Code.CV.anchor_box import MultiBoxPrior


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                                                                     getattr(self, f"cls_{i}"),
                                                                     getattr(self, f"bbox_{i}"))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


# net = TinySSD(num_classes=1)
# X = torch.zeros((32, 3, 256, 256))
# anchors, cls_preds, bboxes = net(X)
#
# print("output anchors", anchors.shape)
# print("output class preds", cls_preds.shape)
# print("output bbox preds", bboxes.shape)


cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


def main():
    import matplotlib
    import matplotlib.pyplot as plt
    import time
    import numpy as np
    from Code.Utils.bbox_dataloader import load_data_pikachu
    from Code.Utils.bboxes import MultiBoxTarget
    matplotlib.use("TkAgg")

    num_epochs = 100
    batch_size = 16
    net = TinySSD(num_classes=1)
    # fig, ax = plt.subplots()
    # ax.set(xlabel="epoch", xlim=[1, num_epochs], xscale="linear", yscale="linear")
    net = net.to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=5e-4)

    train_iter, test_iter = load_data_pikachu(batch_size)
    cls, bbox = [], []
    start_time = time.time()
    for epoch in range(num_epochs):
        cls_evals = 0
        labels = 0
        bbox_evals = 0
        bboxes = 0

        net.train()
        for _iter in train_iter:
            features, target = _iter["image"], _iter["label"]
            trainer.zero_grad()
            # print(features, target)
            X, Y = features.to(device), target.to(device)
            # 生成多尺度锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = MultiBoxTarget(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()

            cls_evals += cls_eval(cls_preds, cls_labels)
            labels += cls_labels.numel()
            bbox_evals += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            bboxes += bbox_labels.numel()

        cls_error, bbox_mae = 1 - cls_evals / labels, bbox_evals / bboxes
        cls.append(cls_error)
        bbox.append(bbox_mae)
        print("epoch %d, loss %.2f" % (epoch, l.mean().data))
    print(f"class error: {cls_error:.2e}, bbox mae: {bbox_mae:.2e}")
    print(f"{len(train_iter.dataset) / (time.time()-start_time):.1f} examples/sec on {str(device)}")
    # ax.plot(np.arange(0, 20, 1), np.array(cls), label="class error")
    # ax.plot(np.arange(0, 20, 1), np.array(bbox), label="bbox mae")
    model_path = os.path.join(ROOT, ".models", "SSD.model")
    if not os.path.isdir(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    torch.save(net, model_path)
    # plt.show()


def eval(net, img_path, device):
    from Code.Utils.bboxes import MultiBoxDetection
    X = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()
    X = X / 256.0

    def predict(X_):
        net.eval()
        anchors, cls_preds, bbox_preds = net(X_.to(device))
        print(bbox_preds[0][:8])
        #print(anchors.shape, cls_preds.shape, bbox_preds.shape)
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = MultiBoxDetection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        return output[0][idx]
    output = predict(X)

    def display(img, output, threshold):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("TkAgg")
        fig = plt.imshow(img)
        for row in output:
            score = float(row[1])
            if score < threshold:
                continue
            h, w = img.shape[0:2]
            bbox = [row[2:6] * torch.tensor([w, h, w, h], dtype=torch.float32, device=device)]
            show_bboxes(fig.axes, bbox, '%.8f' % score, 'w')
        plt.show()
    display(img, output, 0.9)


if __name__ == '__main__':
    # main()
    img = os.path.join(ROOT, "Datasets", "pikachu", "val", "images", "3.png")
    net = torch.load(os.path.join(ROOT, ".models", "SSD.model"))
    eval(net, img, device)