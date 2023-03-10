import os
import sys
sys.path.append(os.getcwd())
from lib.models.common import ConvBNReLU, SPP, BottleneckCSP, Focus, Concat, Detect, Detect_head, Segment_head, C3, InvertedResidual, VoVCSP
from torch.nn import Upsample, ConvTranspose2d, MaxPool2d

from lib.config import cfg

num_det_class = cfg.num_det_class


# n_ = max(round(n * gd=0.33), 1) if n > 1 else n  # depth gain
#  c_ = math.ceil(c * gw / 8) * 8 # channels gain # mobilenet only modify channels

yolov3_tiny = [
    [19],   #Det_out_idx
    # backbone
    [-1, ConvBNReLU, [3, 16, 3, 1]],
    [-1, MaxPool2d, [2, 2]],  # 1

    [-1, ConvBNReLU, [16, 32, 3, 1]],
    [-1, MaxPool2d, [2, 2]],  # 3

    [-1, ConvBNReLU, [32, 64, 3, 1]],
    [-1, MaxPool2d, [2, 2]],  # 5

    [-1, ConvBNReLU, [64, 128, 3, 1]],
    [-1, MaxPool2d, [2, 2]],  # 7

    [-1, ConvBNReLU, [128, 256, 3, 1]],
    [-1, MaxPool2d, [2, 2]],  # 9

    [-1, ConvBNReLU, [256, 512, 3, 1]],
    [-1, MaxPool2d, [1, 1]],  # 11

    # head
    [-1, ConvBNReLU, [512, 1024, 1, 1]],
    [-1, ConvBNReLU, [1024, 256, 1, 1]],   # 13
    [-1, ConvBNReLU, [256, 512, 3, 1]],  # 14

    [-1, ConvBNReLU, [512, 128, 1, 1]],
    [-1, ConvTranspose2d, [128, 128, 4, 2, 1, 0, 128, False]],
    # [-1, Upsample, [None, 2, 'nearest']],
    [[-1, 8], Concat, [1]],
    [-1, ConvBNReLU, [384, 256, 3, 1]],   # 18

    [[14, 18], Detect_head,  [num_det_class, [[23,27,  37,58,  81,82], [81,82,  135,169,  344,319]], [512, 256]]], #Detection head 24

]
