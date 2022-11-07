# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

from pysot.core.config import cfg


class BaseTracker(object):
    """ Base tracker of single objec tracking
    """
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        raise NotImplementedError

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError


class SiameseTracker(BaseTracker):
    def get_im_patch(self, im, pos, model_sz, original_sz, avg_chans, idx, is_first=False, dir="None"):
        sz = original_sz
        im_sz = im.shape  # 第一个参数是高
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))  # 如果context_xmin小于0，表示上一帧的中心点无法满足所在的位置画框不能保证框全部在图形内，此时需要补pad
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))  # 这里也表示第一个参数是高

        context_xmin = context_xmin + left_pad  # 如果此时的left_pad不为0， 那么此处计算的结果将会导致context_xmin为0
        context_xmax = context_xmax + left_pad  # 如果此时的left_pad不为0， 那么此处计算的结果将会导致context_xmax为向左移一个pad
        context_ymin = context_ymin + top_pad  # 同上
        context_ymax = context_ymax + top_pad  # 同上

        r, c, k = im.shape  # r是高，c是宽
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:  # 如果pad不为0，由于刚刚执行了平移的操作，会有0值的出现，此处是将pad处用平均值来代替
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        # if is_first:
        #     cv2.imshow("initial_crop", im_patch)
        #     cv2.waitKey(1)
        # else:
        #     cv2.destroyWindow(str(idx-1) + "_crop" + dir)
        #     # cv2.destroyWindow(str(idx-1) + "initial")
        #     cv2.imshow(str(idx) + "_crop" + dir, im_patch)
        #     cv2.waitKey(0)
        # if idx == 21:
        #     cv2.imshow("search_region_21", im_patch)
        #     cv2.waitKey(1)
        # if idx == 22:
        #     cv2.imshow("search_region_22", im_patch)
        #     cv2.waitKey(1)
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch


    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans, idx, is_first=False, gt_wh=0, dir="center"):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size 255*255
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
            return self.get_im_patch(im, pos, model_sz, original_sz, avg_chans, idx, is_first)
        if is_first:
            return self.get_im_patch(im, pos, model_sz, original_sz, avg_chans, idx, is_first)
        else:
            divid_right = 4
            divid_left = 5
            pos1 = [pos[0] + gt_wh[0] / divid_right, pos[1] + gt_wh[0] / divid_right]
            pos2 = [pos[0] - gt_wh[1] / divid_left, pos[1] - gt_wh[1] / divid_left]
            if dir == "center":
                return self.get_im_patch(im, pos, model_sz, original_sz, avg_chans, idx, is_first,dir)
            elif dir == "left":
                return self.get_im_patch(im, pos2, model_sz, original_sz, avg_chans, idx, is_first,dir)
            elif dir == "right":
                return self.get_im_patch(im, pos1, model_sz, original_sz, avg_chans, idx, is_first,dir)
            else:
                im_patch_0 = self.get_im_patch(im, pos, model_sz, original_sz, avg_chans, idx, is_first,dir)
                im_patch_1 = self.get_im_patch(im, pos1, model_sz, original_sz, avg_chans, idx, is_first,dir)
                im_patch_2 = self.get_im_patch(im, pos2, model_sz, original_sz, avg_chans, idx, is_first,dir)

                return {
                    "im_patch_0": im_patch_0,
                    "im_patch_1": im_patch_1,
                    "im_patch_2": im_patch_2
                }


    '''
    拥有加速度模型的候选帧
    '''
    def get_subwindow_chx(self, im, pos, model_sz, original_sz, avg_chans, acc, pos_x, pos_y):
        """
        args:
            im: bgr based image
            pos: center position 是相对于什么的位置信息（相对于左上角？）
            model_sz: exemplar size
            s_z(original_sz): original size（此处应是CAR当中的1/sqrt（2）倍）
            avg_chans: channel average
            acc: 加速度
            pos_x: 记录之前中点的X值
            pos_y: 记录之前中点的y值
        """
        sz = original_sz
        im_sz = im.shape  # 第一个参数是高
        c = (original_sz + 1) / 2
        if isinstance(pos, float):
            pos = [pos, pos]
        if not len(acc[0]) == 0:
            acc_last_x = acc[0][len(acc[0]) - 1]
            acc_last_y = acc[1][len(acc[1]) - 1]
            acc_now_x = pos_x[len(pos_x) - 1] + pos_x[len(pos_x) - 3] - 2 * pos_x[len(pos_x) - 2]
            acc_now_y = pos_y[len(pos_y) - 1] + pos_y[len(pos_y) - 3] - 2 * pos_y[len(pos_y) - 2]
            acc[0].append(acc_now_x)
            acc[1].append(acc_now_y)
            d_x = (1 - 0.2) * acc_now_x + 0.2 * acc_last_x
            d_y = (1 - 0.2) * acc_now_y + 0.2 * acc_last_y
            if 0 <= pos[0] + d_x <= im_sz[1] and 0 <= pos[1] + d_y <= im_sz[0]:
                pos = [pos[0] + d_x*0.5, pos[1] + d_y*0.5]

        # cv2.rectangle(im, (int(pos[0]-), int(pos[1]-)))

        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))  # 这里也表示第一个参数是高

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape  # r是高，c是宽
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:  # 如果pad不为0，由于刚刚执行了平移的操作，会有0值的出现，此处是将pad处用平均值来代替
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        cv2.imshow("41", im_patch)
        cv2.imshow("1", im)
        cv2.waitKey(0)
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch

    def get_subwindow_ori(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch
