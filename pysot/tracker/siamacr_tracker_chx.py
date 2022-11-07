# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np
import torch.nn.functional as F
import cv2

from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.misc import bbox_clip


class SiamACRTracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamACRTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)  # 生成一个25*25大小的hanning窗
        self.model = model
        self.model.eval()

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
            bbox的四个参数的意思是坐下坐标对， 和 长宽
            centerPos会随着上一帧进行更新？
            size 是跟踪框长宽 也会随着跟踪进行更新
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        self.gt_center_pos = self.center_pos
        self.gt_size = self.size
        self.acc = [[], []]
        # 加速度模型中使用中心点的历史记录
        self.pos_x = [bbox[0]+(bbox[2]-1)/2]
        self.pos_y = [bbox[1]+(bbox[3]-1)/2]

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        # z_crop = self.get_subwindow(img, self.center_pos,
        #                             cfg.TRACK.EXEMPLAR_SIZE,
        #                             s_z, self.channel_average, idx=1, is_first=True)
        z_crop = self.get_subwindow_ori(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def change(self,r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, lrtbs, penalty_lk):  # 这里对数据
        bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
        bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))  # size是GT的宽高
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.  # 目的是什么？
        return disp

    def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)  # 对max_r做限制处理，如果max_r > SCORE_SIZE,取SCORE_SIZE，反正取max(0,max_r)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)
        bbox_region = lrtbs[max_r, max_c, :]
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)  # 这里限制相应框的最大bbox和最小bbox
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)  # 表明最大响应点到左边届的大小
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(p_score_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        p_score_up = p_score_up * mask
        return p_score_up

    def getCenter(self, hp_score_up, p_score_up, scale_score, lrtbs):
        # corse location
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)  # 在【193，193】中的最大的位置
        disp = self.accurate_location(max_r_up, max_c_up)
        disp_ori = disp / self.scale_z
        # todo 添加抑制，抑制跟踪框中心点过分漂移
        # disp_ori[1] = bbox_clip(disp_ori[1], -self.gt_size[0]/3, self.gt_size[0]/3)
        # disp_ori[0] = bbox_clip(disp_ori[0], -self.gt_size[1]/3, self.gt_size[1]/3)

        new_cx = disp_ori[1] + self.center_pos[0]  # 这两个将会更新bbox的中心位置, 是根据最大响应点来调整中心位置
        new_cy = disp_ori[0] + self.center_pos[1]

        # new_cx = self.center_pos[0]
        # new_cy = self.center_pos[1]
        # new_cx = max_r_up
        # new_cy = max_c_up
        return max_r_up, max_c_up, new_cx, new_cy  # 最终得到的是中心点x，y和矫正后的中心点 x1，y1

    def track(self, img, hp, idx, ori=False):
        if ori:
            bbox_ori = self.track_chx(img,hp,idx)
            cx = bbox_ori[0] + bbox_ori[2]/2
            cy = bbox_ori[1] + bbox_ori[3]/2
            width = bbox_ori[2]
            height = bbox_ori[3]
            # update state 中心点和bbox的宽高 中心点是横纵坐标的更新
            self.center_pos = np.array([cx, cy])
            self.size = np.array([width, height])
            # 将跟踪的中心点加入到pos_x 和 pos_y
            self.pos_x.append(cx)
            self.pos_y.append(cy)
            return {
                'bbox': bbox_ori
            }
        else:
            bbox_left = self.track_chx(img,hp,idx,dire="left")
            bbox_right = self.track_chx(img,hp,idx,dire="right")
            factor_left = 1/2
            factor_right = 1/2
            factor_left_xy = 1/2
            factor_right_xy = 1/2

            bbox_refine = [
                           bbox_left[0]*factor_left_xy + bbox_right[0]*factor_right_xy,
                           bbox_left[1]*factor_left_xy + bbox_right[1]*factor_right_xy,
                           bbox_left[2]*factor_left + bbox_right[2]*factor_right,
                           bbox_left[3]*factor_left + bbox_right[3]*factor_right
                           ]
            cx = bbox_refine[0] + bbox_refine[2]/2
            cy = bbox_refine[1] + bbox_refine[3]/2
            width = bbox_refine[2]
            height = bbox_refine[3]
            # update state 中心点和bbox的宽高 中心点是横纵坐标的更新
            self.center_pos = np.array([cx, cy])
            self.size = np.array([width, height])
            # 将跟踪的中心点加入到pos_x 和 pos_y
            self.pos_x.append(cx)
            self.pos_y.append(cy)
            return {
                'bbox_left': bbox_left,
                'bbox_right': bbox_right,
                'bbox_refine': bbox_refine
            }


    '''默认是和原来的CAR一致
    '''
    def track_chx(self, img, hp, idx, dire="center"):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)  # round之后相当于 s_z * 2
        # s_x = s_z * math.sqrt(2)
        # s_x = s_z * 1.2
        # todo 第一张，第二张图，第三张图使用原始的方式解决
        # todo 第四张图开始使用加速度
        if idx == 3:
            pos_x = self.pos_x
            pos_y = self.pos_y
            acc_now_x = pos_x[idx - 1] + pos_x[idx - 3] - 2 * pos_x[idx - 2]
            acc_now_y = pos_y[idx - 1] + pos_y[idx - 3] - 2 * pos_y[idx - 2]
            self.acc[0].append(acc_now_x)
            self.acc[1].append(acc_now_y)
        # x_crop = self.get_subwindow_chx(img, self.center_pos,
        #                             cfg.TRACK.INSTANCE_SIZE,
        #                             round(s_x), self.channel_average, self.acc, self.pos_x, self.pos_y)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average, idx=idx, gt_wh=self.gt_size, dir=dire)
        # x_crop = self.get_subwindow_ori(img, self.center_pos,
        #                                  cfg.TRACK.INSTANCE_SIZE,
        #                                  round(s_x), self.channel_average)

        # x_crop主要作用是将图片切割成统一的大小，切割的中点是上一帧确定的 ,注意，这里将图片裁剪成255*255的大小

        outputs = self.model.track(x_crop)
        # 正式开始跟踪，获得三个分支的分数
        cls = self._convert_cls(outputs['cls']).squeeze()  # 分类的分数只取第一维的数据
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()  # 归一化
        cen = cen.squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()
        #
        upsize = (cfg.TRACK.SCORE_SIZE-1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])   # 这个惩罚项来自论文21
        p_score = penalty * cls * cen  # 这里和论文有些出入
        # p_score = penalty * cls * 1
        if cfg.TRACK.hanming:
            hp_score = p_score*(1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)  # 论文中目标像素点的最大响应值
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)  # cls(i,j)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)  # 未乘以惩罚项的cls
        lrtbs = np.transpose(lrtbs,(1,2,0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE  # 放大倍数
        # get center
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)
        # get w h
        ave_w = (lrtbs_up[max_r_up,max_c_up,0] + lrtbs_up[max_r_up,max_c_up,2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up,max_c_up,1] + lrtbs_up[max_r_up,max_c_up,3]) / self.scale_z
        # 借鉴论文21的惩罚项
        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']  # 更新比例也会进行更新
        new_width = lr*ave_w + (1-lr)*self.size[0]  # bbox的宽度将和前一个的bbox的宽度有关联
        new_height = lr*ave_h + (1-lr)*self.size[1]  # bbox的高度和前一个bbox有关联

        # clip boundary
        cx = bbox_clip(new_cx,0,img.shape[1])
        cy = bbox_clip(new_cy,0,img.shape[0])
        width = bbox_clip(new_width,0,img.shape[1])
        height = bbox_clip(new_height,0,img.shape[0])

        # 绘制最大响应点
        cv2.circle(img, (int(cx), int(cy)), 2, (0, 255, 255), -1)

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return bbox

