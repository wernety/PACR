# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_acr import make_siamacr_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.acr_head import ACRHead
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build car head
        self.acr_head = ACRHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamacr_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

# 获取模板，第一帧为模板，并且没有将模板进行实时的更新
    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        cls, loc, cen, acc = self.acr_head(features)  # 通过call函数回调到网络定义处
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen,
                'acc': acc
                }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()  # 使用permute函数后 接着使用contiguous使内存连续
        cls = F.log_softmax(cls, dim=4)  # 对维度为4做归一化
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        template_gt = data['template_bbox'].cuda()
        search = data['search'].cuda()
        search_gt = data['bbox'].cuda()
        acc = data['search_acc'].cuda()
        acc_gt = data['bbox_acc'].cuda()
        label_cls = data['label_cls'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        af = self.backbone(acc)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
            af = self.neck(af)  # 准备后续进行模板帧的相关相关计算


        features = self.xcorr_depthwise(xf[0],zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)



        cls, loc, cen, accness = self.acr_head(features)

        # cls --> 625*16 == cls_new
        cls_new = F.softmax(cls[:, :, :, :], dim=1).data[:, 1, :, :].permute(1,2,0).reshape(-1, 16)
        # cls的较大一部分点
        cls_max = torch.max(cls_new, 0)[0]  # tuple类型？
        cls_index = cls_new > cls_max * 0.9500
        # 将位置取出
        cls_pos = np.where(cls_index.cpu() == 1)

        # 这里的location是一个常量矩阵，和cls里面的值灭有任何关系，只与形状有关  location的生成是无锚框跟踪的策略
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss, acc_loss = self.loss_evaluator(
            locations,
            cls,
            loc,  # 距离上下左右的距离
            cen, label_cls, search_gt,
            cls_pos=cls_pos,  # 选中的最大响应点
            template_gt=template_gt, acc_gt=acc_gt, accness=accness  # 对应于模板和acc的gt
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss + acc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        outputs['acc_loss'] = acc_loss
        return outputs
