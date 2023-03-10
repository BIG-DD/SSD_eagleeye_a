
import argparse
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import random
from lib.utils import DataLoaderX
import json
import lib.dataset as dataset
from lib.config.default_segment import _C as cfg
from lib.config.default_segment import update_config
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from lib.core.evaluate import SegmentationMetric
from lib.models.YOLOP import YOLOV5S_segment, MCnet_segment_head
from lib.utils.util import time_synchronized
from tqdm import tqdm
import cv2
import numpy as np
from skimage import morphology
import copy
import shutil
import base64
import math
import os

import cv2, torch, argparse
from time import time
import numpy as np
from torch.nn import functional as F

from models import UNet
from utils import utils
import copy



def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    parser.add_argument('--finetune',
                        default='',
                        type=str,
                        help='log directory')
    parser.add_argument('--sparsity_factor',
                        type=int,
                        default=1e-4,
                        help='model directory')
    parser.add_argument('--prune_scale_rate',
                        type=int,
                        default=0.5,
                        help='model directory')
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args


def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    print("begin to load data")
    # Data loading
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
        mean=[0.0, 0.0, 0.0], std=[0.0, 0.0, 0.0]
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET + '_segment')(
        cfg=cfg,
        is_train=False,
        inputsize=[256, 256],
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    BATCH_SIZE = 1
    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=BATCH_SIZE,  # * len(cfg.GPUS)
        shuffle=False,
        num_workers=0, # 8
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset_segment.collate_fn
    )
    print('load data finished')

    save_path = 'D:/project/python/yolop/runs/parking_data_segment/new_part3/out/'  #改这里
    model_path = r"D:\model\parking_slot\0527_160351\69.pth"  #改这里
    # set all the configurations

    print("begin to load data")
    # Data loading

    # bulid up model
    # start_time = time.time()
    print("begin to bulid up model...")

    # device = 'cuda'
    # num_det_class = 4
    # print("load model to device")
    # model = MCnet_segment_head(num_det_class, YOLOV5S_segment)
    # # print("load finished")
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # model = model.to('cuda')
    # # print("finish build model")

    model = UNet(
        backbone="mobilenetv2",
        num_classes=2,
        pretrained_backbone=None
    )
    model = model.cuda()
    trained_dict = torch.load(model_path, map_location="cpu")['state_dict']
    model.load_state_dict(trained_dict, strict=False)
    model.eval()
    device = 'cuda'
    model.gr = 1.0
    model.nc = 2
    train_val = cfg.DATASET.TEST_SET+'/'
    test_segment(model, valid_dataset, valid_loader, save_path, device, train_val)

def test_segment(model, valid_dataset, val_loader, save_path, device, train_val):
    ll_metric = SegmentationMetric(2)  # segment confusion matrix
    # switch to train mode
    model.eval()
    data_len = len(val_loader)
    for idx, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        # sys.stdout.write('\r>>val  iter %d -> %d ' % (data_len, i + 1))
        # sys.stdout.flush()
        data = valid_dataset.db[idx]
        img_file = data["images"]
        label_file = data["lane"]
        _, name = os.path.split(img_file)
        json_file = 'D:/data/part1/' + name.replace('.jpg', '.json') #改这里
        if not os.path.exists(json_file):
            continue

        # if os.path.exists('/media/z590/D/PublicDataSet/BYD_parking_slot/parking_data/error_train/'+name):
        #     continue

        img = img.to(device, non_blocking=True)
        assign_target = []
        for tgt in target:
            assign_target.append(tgt.to(device))
        target = assign_target
        nb, _, height, width = img.shape    # batch size, channel, height, width

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            # print(pad_w, pad_h, height, width)
            ratio = shapes[0][1][0][0]

            t = time_synchronized()
            ll_seg_out = model(img)
            t_inf = time_synchronized() - t
            # lane line segment evaluation
            _, ll_predict = torch.max(ll_seg_out, 1)
            _, ll_gt = torch.max(target[0], 1)
            # print(ll_predict.shape)
            # print(ll_gt.shape)

            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()

            # 把预测结构做开操作，然后提前骨架，在膨胀。在与标注图片相加，提取骨架，在膨胀。
            ll_seg_mask = ll_seg_out[0][:, pad_h:height - pad_h, pad_w:width - pad_w].unsqueeze(0)
            ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=float(1 / ratio),
                                                          mode='bilinear')
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)

            ll_gt_mask = target[0][0][:, pad_h:height - pad_h, pad_w:width - pad_w].unsqueeze(0)
            ll_gt_mask = torch.nn.functional.interpolate(ll_gt_mask, scale_factor=float(1 / ratio),
                                                         mode='bilinear')
            _, ll_gt_mask = torch.max(ll_gt_mask, 1)
            ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()
            ll_gt_mask = np.array(ll_gt_mask * 255, dtype='uint8')
            # print(ll_gt_mask.shape)

            ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
            ll_seg_mask = np.array(ll_seg_mask * 255, dtype='uint8')
            print(ll_seg_mask)

            #seg与gt做iou计算，从某个阈值分开。
            # if ll_IoU > 0.7:
            #     f = ll_IoU
            #     shutil.copy(img_file, 'D:/project/python/yolop/runs/parking_data_segment/new_part3/water_out_0.7/' + name)
            #     shutil.copy(json_file, 'D:/project/python/yolop/runs/parking_data_segment/new_part3/water_out_0.7/' + name.replace('.jpg', '.json'))
            #     cv2.imwrite('D:/project/python/yolop/runs/parking_data_segment/new_part3/water_out_0.7/' + name.split('.jpg')[0] + '_seg_%.3f_' % f + '.png', ll_seg_mask)
            #     cv2.imwrite(
            #         'D:/project/python/yolop/runs/parking_data_segment/new_part3/water_out_0.7/' + name.split('.jpg')[0] + '_gt_' + '.png',
            #         ll_gt_mask)
            # else:
            #     f = ll_IoU
            #     shutil.copy(img_file, 'D:/project/python/yolop/runs/parking_data_segment/new_part3/water_in_0.7/' + name)
            #     shutil.copy(json_file, 'D:/project/python/yolop/runs/parking_data_segment/new_part3/water_in_0.7/' + name.replace('.jpg', '.json'))
            #     cv2.imwrite('D:/project/python/yolop/runs/parking_data_segment/new_part3/water_in_0.7/' + name.split('.jpg')[0] + '_seg_%.3f_' % f + '.png',
            #                 ll_seg_mask)
            #     cv2.imwrite(
            #         'D:/project/python/yolop/runs/parking_data_segment/new_part3/water_in_0.7/' + name.split('.jpg')[0] + '_gt_' + '.png',
            #         ll_gt_mask)

            # ll_seg_mask = ll_seg_mask[:, 1:257]
            # print(ll_seg_mask.shape)

if __name__ == '__main__':
    main()
