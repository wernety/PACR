"""
This file contains specific functions for computing losses of SiamAcr
file
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


INF = 100000000

'''在select的位置求pred和label的损失
    整篇代码上只有两个地方使用该损失'''
def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)   # 计算label当中为1的位置，并将这些位置映射到cls（pred，【16*25*25，2】）上，计算损失
    loss_neg = get_cls_loss(pred, label, neg)   # 同上，但是选择label为0的位置
    return loss_pos * 0.5 + loss_neg * 0.5      # 取两位的平均


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


'''该损失函数用作reg的损失'''
class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()
        self.border_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]  # 表示对应的点离左边界的距离
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)  #
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect  # 计算有问题
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

class GIOU_LOSS(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()
        self.border_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, pred_cen, target_cen, weight=None):
        pred_left = pred[:, 0]  # 表示对应的点离左边界的距离
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)  #
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)
        area_intersect = w_intersect * h_intersect  # 计算有问题
        area_union = target_aera + pred_aera - area_intersect

        d_pow = pow((target_cen[0] - pred_cen[0]), 2) + pow((target_cen[1] - pred_cen[1]))
        c_pow = pow(torch.max(pred_left, target_left) + torch.max(pred_right, target_right), 2) + \
                pow(torch.max(target_bottom, pred_bottom), torch.max(target_top, pred_top), 2)

        d_iou = area_intersect/area_union - d_pow/c_pow
        losses = d_iou

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class SiamACRLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self, cfg):
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def prepare_targets(self, points, labels, gt_bbox, cls_pos):

        labels, reg_targets, cls_pos_new = self.compute_targets_for_locations(
            points, labels, gt_bbox, cls_pos
        )

        return labels, reg_targets, cls_pos_new

    def prepare_targets_acc(self, locations, template_gt, acc_gt):

        xs, ys = locations[:, 0], locations[:, 1]
        bboxes_template = template_gt
        bboxes_acc = acc_gt
        l_template = xs[:, None] - bboxes_template[:, 0][None].float()  # 只是将bbox的值做了一个线性变换，表示每个点到
        t_template = ys[:, None] - bboxes_template[:, 1][None].float()  # 算出模板
        r_template = bboxes_template[:, 2][None].float() - xs[:, None]
        b_template = bboxes_template[:, 3][None].float() - ys[:, None]
        template_targets_per_im = torch.stack([l_template, t_template, r_template, b_template], dim=2)  # 将线性变换的值按层叠加

        l_acc = xs[:, None] - bboxes_acc[:, 0][None].float()  # 只是将bbox的值做了一个线性变换，表示每个点到
        t_acc = ys[:, None] - bboxes_acc[:, 1][None].float()  # 算出模板
        r_acc = bboxes_acc[:, 2][None].float() - xs[:, None]
        b_acc = bboxes_acc[:, 3][None].float() - ys[:, None]
        acc_targets_per_im = torch.stack([l_acc, t_acc, r_acc, b_acc], dim=2)  # 将线性变换的值按层叠加

        return template_targets_per_im.permute(1,0,2).contiguous(), acc_targets_per_im.permute(1,0,2).contiguos()

    def compute_targets_for_locations(self, locations, labels, gt_bbox, cls_pos):
        # reg_targets = []
        # locations是一组特殊的值 【【32，40，..32.... 232】【32，32，...，32，32，...232，232】】
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE**2,-1)  #

        l = xs[:, None] - bboxes[:, 0][None].float()  # 维度是 625 * 16
        t = ys[:, None] - bboxes[:, 1][None].float()  # 算出模板
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)  # 将线性变换的值按层叠加

        s1 = reg_targets_per_im[:, :, 0] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()  # 维度是625 * 16
        s2 = reg_targets_per_im[:, :, 2] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        # zy = is_in_boxes.cpu().numpy()
        pos = np.where(is_in_boxes.cpu() == 1)
        # todo 计算最大响应点在bbox的值
        # labels[cls_pos] = 1  # 维度是625 * 16
        cross_pos_labels = np.zeros_like(labels.cpu())
        cross_pos_labels[cls_pos] = 1
        cross_pos_labels[pos] = cross_pos_labels[pos] + 1
        cross_pos = np.where(cross_pos_labels == 2)

        labels[pos] = 1
        # labels[cross_pos] = 1
        return labels.permute(1,0).contiguous(), reg_targets_per_im.permute(1,0,2).contiguous(), cross_pos  # 第二个维度是16 * 625 * 4

    '''centerness为reg的第0，2维度的最小值除以0，2维度的最大值然后乘以 （1，3维度上的最小值除以1，3维度上的最大值）然后开平方得到的结果
    '''
    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        centerness = torch.where(centerness<0, 0, centerness)
        return torch.sqrt(centerness)

    # 计算平均加速度分数
    def copmute_accness_targets(self, locations, search_gt, template_gt, acc_gt, cls_pos):
        xs, ys = locations[:, 0], locations[:, 1]
        bboxes_template = template_gt
        bboxes_acc = acc_gt
        bboxes_search = search_gt
        l_template = xs[:, None] - bboxes_template[:, 0][None].float()  # 维度为625 * 16
        l_acc = xs[:, None] - bboxes_acc[:, 0][None].float()
        l_search = xs[:, None] - bboxes_search[:, 0][None].float()
        acc_x_temp = l_search + l_template - 2*l_acc  # 维度为625 * 16
        # acc_avg = acc_x_temp[cls_pos].sum()/cls_pos[0].size  # 维度为 1， 此时为16张图的平均值
        self.caculate_per_image_ave(acc_x_temp, cls_pos)
        acc_x = self.acc_norm(acc_x_temp)

        t_template = ys[:, None] - bboxes_template[:, 1][None].float()
        t_acc = ys[:, None] - bboxes_acc[:, 1][None].float()
        t_search = ys[:, None] - bboxes_search[:, 1][None].float()
        acc_y_temp = t_search + t_template - 2*t_acc
        self.caculate_per_image_ave(acc_y_temp, cls_pos)
        acc_y = self.acc_norm(acc_y_temp)
        # todo 方案一 直接除以2
        # acc_score = acc_x*acc_y*0.5
        # acc_score[cls_pos] = acc_score[cls_pos] * 2
        # todo 方案二 直接置0
        acc_score = acc_x*acc_y
        acc_score_temp = torch.zeros_like(acc_score)
        acc_score_temp[cls_pos] = acc_score[cls_pos]
        acc_score = acc_score_temp
        return acc_score.permute(1, 0).contiguous()   # 返回的是(625,16)

    def caculate_per_image_ave(self, acc_temp, cls_pos):
        col = acc_temp.size(1)
        d = {}
        # 计算625*16中的每一列的平均值 16是batch_size
        for i in range(16):
            # 计算每一张图片的最大相应区间点集的位置
            index = np.where(cls_pos[1] == i)
            index_x_temp = cls_pos[0][index[0]]
            index_y_temp = cls_pos[1][index[0]]
            index_temp = (index_x_temp, ) + (index_y_temp, )
            d[i] = index_temp
            # 计算平均值
            avg = acc_temp[index_temp].sum() / acc_temp[index_temp].size(0)
            # 将acc_temp对应的列除以平均值 然后
            acc_temp[index_temp] = acc_temp[index_temp] / avg

    def acc_norm(self, acc_temp):
        acc_temp = torch.where(acc_temp<0,0,acc_temp)
        acc_temp = torch.where(acc_temp>1,1/(acc_temp*10),acc_temp)
        return acc_temp

    def __call__(self, locations, box_cls, box_regression, centerness, labels, search_gt, cls_pos, template_gt, acc_gt, accness):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])
            cls_pos 根据cls最大响应值选择的位置，用以计算损失

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        global acc_loss
        search_gt_template = search_gt
        label_cls, search_gt, cls_pos = self.prepare_targets(locations, labels, search_gt, cls_pos)  # 这里的 labels 和 reg_targets 都是 GT 的变形（线性变化）
        accness_target_ori = self.copmute_accness_targets(locations, search_gt_template, template_gt, acc_gt, cls_pos)  # 出来的分数是625*16, 是加速度分数
        # template_gt, acc_gt = self.prepare_targets_acc(locations, template_gt, acc_gt)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))
        search_gt_flatten = (search_gt.view(-1, 4))
        centerness_flatten = (centerness.view(-1))
        accness_faltten = accness.view(-1)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)   # pos_index只与BBox有关，选取正样本的位置

        box_regression_flatten = box_regression_flatten[pos_inds]  # 这句的作用是取对应于GT中是目标的点的值
        search_gt_flatten = search_gt_flatten[pos_inds]  # 在回归框中选择对应于GT中是目标的点的值
        centerness_flatten = centerness_flatten[pos_inds]  # 取对应于中心分数点的GT的值
        accness_faltten_pos = accness_faltten[pos_inds]  # 正样本的加速度分数
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)  # labels_flatten的维度是【16*25*25，2】，计算cls损失熵的方法是，在cls中分别取label中为1和0的位置，然后计算损失，将两个损失平均相加

        if pos_inds.numel() > 0:
            # centerness_targets = self.compute_centerness_targets(search_gt_flatten)  # 计算cen的分数，是用GT进行计算，反映的是在GT中处于中心的程度
            centerness_targets = self.compute_centerness_targets(search_gt_flatten)  # 计算acc的分数，是用GT进行计算，反映的是在GT中处于中心的程度
            accness_target = accness_target_ori.view(-1)  # 选取正样本的加速度进行回归
            accness_target = accness_target[pos_inds]
            # accness_target = accness_target_ori.view(-1)

            weight_target = centerness_targets + accness_target

            reg_loss = self.box_reg_loss_func(              # 计算reg的损失
                box_regression_flatten,
                search_gt_flatten,
                # centerness_targets
                weight_target
            )
            centerness_loss = self.centerness_loss_func(    # 计算cen的损失分数
                centerness_flatten,
                centerness_targets
            )
            acc_loss = self.centerness_loss_func(
                accness_faltten[pos_inds],
                accness_target
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss, acc_loss


def make_siamacr_loss_evaluator(cfg):
    loss_evaluator = SiamACRLossComputation(cfg)
    return loss_evaluator
