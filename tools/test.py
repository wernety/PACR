# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys
import shutil

current_file = os.path.dirname(__file__)
os.chdir(current_file)
sys.path.append('.')
sys.path.append('../')

from pysot.core.config import cfg
from pysot.tracker.siamacr_tracker_chx import SiamACRTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder

from toolkit.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='OTB100',
        help='datasets')#OTB100 LaSOT UAV123 GOT-10k
# parser.add_argument('--dataset', type=str, default='LaSOT',
#         help='datasets')#OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_false', default=True,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='snapshot/checkpoint_e10.pth',help='snapshot of models to eval')
# parser.add_argument('--snapshot', type=str, default='snapshot/model_general.pth',help='snapshot of models to eval')

# parser.add_argument('--config', type=str, default='../experiments/siamAcr_r50/config.yaml',
#         help='config file')
parser.add_argument('--config', type=str, default='../experiments/siamAcr_r50/config.yaml',
        help='config file')

args = parser.parse_args()

torch.set_num_threads(1)


def main():
    # load config
    cfg.merge_from_file(args.config)

    # hp_search
    params = getattr(cfg.HP_SEARCH, args.dataset)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # dataset_root = os.path.join(cur_dir, 'H:\\lasot_test', args.dataset)
    dataset_root = os.path.join(cur_dir, 'H:\\', args.dataset)
    # dataset_root = os.path.join('H:\\', args.dataset)
    print(dataset_root)

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamACRTracker(model, cfg.TRACK)

    # create dataset
    # dataset = DatasetFactory.create_dataset(name=args.dataset,
    #                                         dataset_root=dataset_root,
    #                                         load_img=False,
    #                                         is_test=True)

    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-2] + str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])

    flag = True

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        # if video.name != 'airplane-15' and flag:
        #     continue
        # if video.name == 'airplane-15':
        #     flag = False
        #     continue
        toc = 0
        pred_bboxes = []
        track_times = []
        pred_bboxs_refine = []
        for idx, (img, gt_bbox) in enumerate(video):
            cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])),
                          (int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])),
                          (0, 255, 255), 3)
            cv2.imshow('1', img)
            cv2.waitKey(0)
            # CAR 跟踪器
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))  # 初始化第一帧的时候，将GT转换成BBox GT就是左上和宽高
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img, hp, idx, ori=True)
                pred_bbox = outputs['bbox']
                # pred_bbox_left = outputs['bbox_left']
                # pred_bbox_right = outputs['bbox_right']
                pred_bboxes.append(pred_bbox)

            # 我的跟踪器
            # tic = cv2.getTickCount()
            # if idx == 0:
            #     cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))  # 初始化第一帧的时候，将GT转换成BBox GT就是左上和宽高
            #     gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
            #     tracker.init(img, gt_bbox_)
            #     pred_bbox_refine = gt_bbox_
            #     pred_bboxs_refine.append(pred_bbox_refine)
            # else:
            #     outputs = tracker.track(img, hp, idx)
            #     # pred_bbox_left = outputs['bbox_left']
            #     # pred_bbox_right = outputs['bbox_right']
            #     pred_bbox_refine = outputs['bbox_refine']
            #     pred_bboxs_refine.append(pred_bbox_refine)
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            # track_times.append("(cv2.getTickCount() - tic)/cv2.getTickFrequency()")
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0 and True:
                if not any(map(math.isnan,gt_bbox)):
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    # pred_bbox_left = list(map(int, pred_bbox_left))
                    # pred_bbox_right = list(map(int, pred_bbox_right))
                    # pred_bbox_refine = list(map(int, pred_bbox_refine))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    # cv2.rectangle(img, (gt_bbox[0]+1, gt_bbox[1]+1),
                    #               (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 255), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    # cv2.rectangle(img, (pred_bbox_left[0], pred_bbox_left[1]),
                    #               (pred_bbox_left[0] + pred_bbox_left[2], pred_bbox_left[1] + pred_bbox_left[3]), (49,101,95), 3)
                    # cv2.rectangle(img, (pred_bbox_right[0], pred_bbox_right[1]),
                    #               (pred_bbox_right[0] + pred_bbox_right[2], pred_bbox_right[1] + pred_bbox_right[3]), (188,175,100), 3)
                    # cv2.rectangle(img, (pred_bbox_refine[0], pred_bbox_refine[1]),
                    #               (pred_bbox_refine[0] + pred_bbox_refine[2], pred_bbox_refine[1] + pred_bbox_refine[3]), (188,175,100), 3)
                    # cv2.circle(img, (int(pred_bbox[0] + pred_bbox[2]/2), int(pred_bbox[1] + pred_bbox[3]/2)), 2, (0, 0, 255), -1)  # 取中心点计算加速度
                    # print('==============')
                    # print(int(pred_bbox[0] + pred_bbox[2]/2))
                    # print(int(pred_bbox[1] + pred_bbox[3]/2))
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # todo 如果想改变结果存储的位置，在这儿改变
        # model_path_car = os.path.join('results', args.dataset, model_name + "chx_" + "direct_combine" + "_vid_got10_e19")
        model_path_car = os.path.join('results', args.dataset, model_name + "chx" + "e11=33" + 'dtb-otb')
        if not os.path.isdir(model_path_car):
            os.makedirs(model_path_car)
        result_path_car = os.path.join(model_path_car, '{}.txt'.format(video.name))

        with open(result_path_car, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')

        # model_path = os.path.join('results', args.dataset, model_name + "chx")
        # if not os.path.isdir(model_path):
        #     os.makedirs(model_path)
        # result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        #
        # # 将结果写入到文件中
        # with open(result_path, 'w') as f:
        #     for x in pred_bboxs_refine:
        #         f.write(','.join([str(i) for i in x])+'\n')
        #
        #
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))
    # os.chdir(model_path)
    # os.chdir(model_path_car)
    # save_file = 'I:\\tracker\\SiamCAR-master\\tools\\results'
    # shutil.make_archive(save_file, 'zip')
    # print('Records saved at', save_file + '.zip')


if __name__ == '__main__':
    main()
