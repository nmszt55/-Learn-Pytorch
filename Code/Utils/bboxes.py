import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


def bbox_to_rect(bbox, color):
    # 将边界框模式(左上x,左上y,右下x,右下y)转换成matplotlib格式:
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
                         fill=False, edgecolor=color, linewidth=2)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有锚框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj=default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ["b", "g", "r", "m", "c"])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = "k" if color == "w" else "w"
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va="center", ha="center", fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


def compute_intersection(set_1, set_2):
    """
    计算Anchor之间的交集
    anchor表示成(xmin, ymin, xmax, ymax)
    """
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def compute_jaccard(set_1, set_2):
    """计算Anchor之间的jaccard系数"""
    intersection = compute_intersection(set_1, set_2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection
    return intersection / union


def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    Params:
        bb: 真实边界框,shape:[nb, 4]
        anchor: 待分配的锚框,shape[na, 4]
        jaccard_threshold: 交并比阈值
    Return:
        assigned_idx: shape:(na, )，每个anchor分配的真实标签索引，如果未分配则为-1
    """
    na = anchor.shape[0]
    nb = bb.shape[0]
    jaccard = compute_jaccard(anchor, bb).detach().cpu().numpy()
    assign_idx = np.ones(na) * -1

    jaccard_cp = jaccard.copy()
    for j in range(nb):
        i = np.argmax(jaccard_cp[:, j])
        assign_idx[i] = j
        jaccard_cp[i, :] = float("-inf")  # 赋值为负无穷，相当于去掉这一行

    # 处理还未被分配的anchor,要求满足jaccard threshold
    for i in range(na):
        if assign_idx[i] == -1:
            j = np.argmax(jaccard[i, :])
            if jaccard[i, j] >= jaccard_threshold:
                assign_idx[i] = j
    return torch.tensor(assign_idx, dtype=torch.long)


def xy_to_cxcy(xy):
    """将(左上x,左上y,右下x,右下y)改成(中心x,中心y,w,h)的形式"""
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # cx, cy
                      xy[:, 2:] - xy[:, :2]], 1)   # w, h


def MultiBoxTarget(anchor, label):
    """为锚框标注类别和偏移量"""
    assert len(anchor.shape) == 3 and len(label.shape) == 3
    bn = label.shape[0]
    device = anchor.device

    def MultiBoxTarget_one(anc, lab, eps=1e-6):
        an = anc.shape[0]
        assigned_idx = assign_anchor(lab[:, 1:], anc)
        bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4).to(device)

        cls_labels = torch.zeros(an, dtype=torch.long, device=device)
        # 所有anchor对应的bb坐标
        assigned_bb = torch.zeros((an, 4), dtype=torch.float32, device=device)
        for i in range(an):
            bb_idx = assigned_idx[i]
            # 非背景
            if bb_idx >= 0:
                cls_labels[i] = lab[bb_idx, 0].long().item() + 1
                assigned_bb[i, :] = lab[bb_idx, 1:]

        center_anc = xy_to_cxcy(anc)
        center_assigned_bb = xy_to_cxcy(assigned_bb)

        offset_xy = 10.0 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
        offset_wh = 5.0 * torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])

        offset = torch.cat([offset_xy, offset_wh], dim=1) * bbox_mask

        return offset.view(-1), bbox_mask.view(-1), cls_labels

    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    for b in range(bn):
        offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0, :, :], label[b, :, :])

        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)
    return [bbox_offset, bbox_mask, cls_labels]


from collections import namedtuple
Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])


def non_max_suppression(bb_info_list, nms_threshold=0.5):
    """
    非极大抑制处理边界框
    Params:
        bb_info_list: 信息列表，包含置信度，预测类别等信息
    return:
        output: bb_info_list的列表，只保留过滤后的边框信息
    """
    output = []
    sorted_bb_info_list = sorted(bb_info_list, key=lambda x: x.confidence, reverse=True)
    while len(sorted_bb_info_list) != 0:
        best = sorted_bb_info_list.pop(0)
        output.append(best)

        if len(sorted_bb_info_list) == 0:
            break

        bb_xyxy = []
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)

        iou = compute_jaccard(torch.tensor([best.xyxy]),
                              torch.tensor(bb_xyxy))[0]
        n = len(sorted_bb_info_list)
        sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] < nms_threshold]
    return output


def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold=0.5):
    assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
    bn = cls_prob.shape[0]

    def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold):
        pred_bb_num = c_p.shape[1]
        anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy()

        confidence, class_id = torch.max(c_p, 0)
        confidence = confidence.detach().cpu().numpy()

        pred_bb_info = [Pred_BB_Info(
            index=i,
            class_id=class_id[i] - 1,
            confidence=confidence[i],
            xyxy=[*anc[i]]
        ) for i in range(pred_bb_num)]

        obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]

        output = []
        for bb in pred_bb_info:
            output.append([
                (bb.class_id if bb.index in obj_bb_idx else -1.0),
                bb.confidence,
                *bb.xyxy
            ])
        return torch.tensor(output)

    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))

    return torch.stack(batch_output)
