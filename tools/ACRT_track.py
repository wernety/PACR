from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import vot
import time
import numpy
import collections
import argparse
import os
import cv2
import torch
import numpy as np
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
    handle = vot.VOT("rectangle")
    selection = handle.region()

    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = cv2.imread(imagefile)

    cfg.merge_from_file(args.config)

    params = getattr(cfg.HP_SEARCH, args.dataset)
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    ACRT = SiamACRTracker(model, cfg.TRACK)
    model_name = args.snapshot.split('/')[-2] + str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])

    # init Tracker
    idx = 0
    gt_bbox = [selection.x, selection.y, selection.width, selection.height]
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))  # 初始化第一帧的时候，将GT转换成BBox GT就是左上和宽高
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    ACRT.init(image, gt_bbox_)
    # cv2.rectangle(image, (int(selection.x), int(selection.y)),
    #               (int(selection.x + selection.width), int(selection.y + selection.height))
    #               , (0, 255, 255), 3)
    # cv2.rectangle(image, (int(gt_bbox_[0]), int(gt_bbox_[1])),
    #               (int(gt_bbox_[0] + gt_bbox_[2]), int(gt_bbox_[1] + gt_bbox_[3])),
    #               (0, 255, 255), 3)
    # cv2.imshow('1', image)
    # cv2.waitKey(2 * 1000)
    # print(selection, image)
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.imread(imagefile)
        region = ACRT.track(image, hp, idx=idx, ori=True)
        # cv2.imshow('pre', image)
        # cv2.waitKey(0)
        # print(region['bbox'])
        # cv2.rectangle(image, (int(region['bbox'][0]), int(region['bbox'][1])),
        #               (int(region['bbox'][0] + region['bbox'][2]), int(region['bbox'][1] + region['bbox'][3])),
        #               (0, 255, 0), 3)
        # cv2.imshow('aft', image)
        # cv2.waitKey(1)
        idx = idx + 1
        region_vot = vot.Rectangle(region['bbox'][0], region['bbox'][1], region['bbox'][2], region['bbox'][3])
        handle.report(region_vot)

main()

