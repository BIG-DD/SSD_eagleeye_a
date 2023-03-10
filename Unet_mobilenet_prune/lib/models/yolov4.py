import os
import sys
sys.path.append(os.getcwd())
from lib.models.common import ConvBNReLU_mask, Concat, Detect_head, VoV_mask, VoVCSP_mask, Bottleneck_mask
from lib.models.common import ConvBNReLU, VoV, VoVCSP
from torch.nn import Upsample, ConvTranspose2d, MaxPool2d
# from lib.config import cfg
from lib.config.default_detect import _C as cfg
num_det_class = cfg.num_det_class

# ConvTranspose2d:TDA2 group = channels
# n_ = max(round(n * gd=0.33), 1) if n > 1 else n  # depth gain
#  c_ = math.ceil(c * gw / 8) * 8 # channels gain # mobilenet only modify channels

yolov4_tiny_origin = [
    [21],   #Det_out_idx
    # backbone
    [-1, ConvBNReLU, [3, 32, 3, 2]],  # 0-P1/2
    [-1, ConvBNReLU, [32, 64, 3, 2]],  # 1-P2/4

    [-1, ConvBNReLU, [64, 64, 3, 1]],
    [-1, VoV, [64, 64]],
    [[-1, 2], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 5-P3/8

    [-1, ConvBNReLU, [128, 128, 3, 1]],
    [-1, VoV, [128, 128]],
    [[-1, 6], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 9-P4/16

    [-1, ConvBNReLU, [256, 256, 3, 1]],
    [-1, VoV, [256, 256]],
    [[-1, 10], Concat, [1]],
    [-1, MaxPool2d, [2, 2]],  # 13-P5/32

    [-1, ConvBNReLU, [512, 512, 3, 1]],  # 14
    # head
    [-1, ConvBNReLU, [512, 256, 1, 1]],
    [-1, ConvBNReLU, [256, 512, 3, 1]],   # 16

    [-1, ConvBNReLU, [512, 256, 1, 1]],    # 17
    [-1, ConvTranspose2d, [256, 256, 4, 2, 1, 0, 256, False]],  #
    # [-1, Upsample, [None, 2, 'nearest']],
    [[-1, 11], Concat, [1]],
    [-1, ConvBNReLU, [512, 256, 3, 1]],   # 20

    [[20, 16], Detect_head,  [num_det_class, [[10,14,  23,27,  37,58], [81,82,  135,169,  344,319]], [256, 512]]], #Detection head 24

]


yolov4_tiny = [
    [19],   #Det_out_idx
    # backbone
    [-1, ConvBNReLU, [3, 32, 3, 2]],  # 0-P1/2
    [-1, ConvBNReLU, [32, 64, 3, 2]],  # 1-P2/4

    [-1, ConvBNReLU, [64, 64, 3, 1]],
    [-1, VoVCSP, [64, 64]],
    # [[-1, 2], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 4-P3/8

    [-1, ConvBNReLU, [128, 128, 3, 1]],
    [-1, VoVCSP, [128, 128]],
    # [[-1, 6], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 7-P4/16

    [-1, ConvBNReLU, [256, 256, 3, 1]],
    [-1, VoV, [256, 256]],
    [[-1, 8], Concat, [1]],
    [-1, MaxPool2d, [2, 2]],  # 11-P5/32

    [-1, ConvBNReLU, [512, 512, 3, 1]],  # 12
    # head
    [-1, ConvBNReLU, [512, 256, 1, 1]],
    [-1, ConvBNReLU, [256, 512, 3, 1]],   # 14

    [-1, ConvBNReLU, [512, 256, 1, 1]],    # 15
    [-1, ConvTranspose2d, [256, 256, 4, 2, 1, 0, 256, False]],  #
    # [-1, Upsample, [None, 2, 'nearest']],
    [[-1, 9], Concat, [1]],
    [-1, ConvBNReLU, [512, 256, 3, 1]],   # 18

    [[18, 14], Detect_head,  [num_det_class, [[10,14,  23,27,  37,58], [81,82,  135,169,  344,319]], [256, 512]]], #Detection head 24

]

# use train model
yolov4_tiny_mask = [
    [19],   #Det_out_idx
    # backbone
    [-1, ConvBNReLU_mask, [3, 32, 3, 2, 1]],  # 0-P1/2
    [-1, ConvBNReLU_mask, [32, 64, 3, 2, 1]],  # 1-P2/4

    [-1, ConvBNReLU_mask, [64, 64, 3, 1, 1]],
    [-1, VoVCSP_mask, [64, 64]],
    # [[-1, 2], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 4-P3/8

    [-1, ConvBNReLU_mask, [128, 128, 3, 1, 1]],
    [-1, VoVCSP_mask, [128, 128]],
    # [[-1, 6], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 7-P4/16

    [-1, ConvBNReLU_mask, [256, 256, 3, 1, 1]],
    [-1, VoV_mask, [256, 256]],
    [[-1, 8], Concat, [1]],
    [-1, MaxPool2d, [2, 2]],  # 11-P5/32

    [-1, ConvBNReLU_mask, [512, 512, 3, 1, 1]],  # 12
    # head
    [-1, ConvBNReLU_mask, [512, 256, 1, 1, 0]],
    [-1, ConvBNReLU_mask, [256, 512, 3, 1, 1]],   # 14

    [-1, ConvBNReLU_mask, [512, 256, 1, 1, 0]],    # 15
    [-1, ConvTranspose2d, [256, 256, 4, 2, 1, 0, 256, False]],  #
    # [-1, Upsample, [None, 2, 'nearest']],
    [[-1, 9], Concat, [1]],
    [-1, ConvBNReLU_mask, [512, 256, 3, 1, 1]],   # 18

    [[18, 14], Detect_head,  [num_det_class, [[10,14,  23,27,  37,58], [81,82,  135,169,  344,319]], [256, 512]]], #Detection head 24

]


yolov4_tiny_mask_advance = [
    [18],   #Det_out_idx
    # backbone
    [-1, ConvBNReLU_mask, [3, 32, 3, 2, 1]],  # 0-P1/2
    [-1, ConvBNReLU_mask, [32, 64, 3, 2, 1]],  # 1-P2/4

    [-1, ConvBNReLU_mask, [64, 64, 3, 1, 1]],
    [-1, VoVCSP_mask, [64, 64]],
    # [[-1, 2], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 4-P3/8

    [-1, ConvBNReLU_mask, [128, 128, 3, 1, 1]],
    [-1, VoVCSP_mask, [128, 128]],
    # [[-1, 6], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 7-P4/16

    [-1, ConvBNReLU_mask, [256, 256, 3, 1, 1]],    # 8
    # [-1, VoV_mask, [256, 256]],
    # [[-1, 8], Concat, [1]],
    [-1, VoVCSP_mask, [256, 256]],      # 9

    [-1, MaxPool2d, [2, 2]],  # 10-P5/32

    [-1, ConvBNReLU_mask, [512, 512, 3, 1, 1]],  # 11
    # head
    [-1, ConvBNReLU_mask, [512, 256, 1, 1, 0]],
    [-1, ConvBNReLU_mask, [256, 512, 3, 1, 1]],   # 13

    [-1, ConvBNReLU_mask, [512, 256, 1, 1, 0]],    # 14
    [-1, ConvTranspose2d, [256, 256, 4, 2, 1, 0, 256, False]],  #
    # [-1, Upsample, [None, 2, 'nearest']],
    [[-1, 9], Concat, [1]],
    [-1, ConvBNReLU_mask, [512+256, 256, 3, 1, 1]],   # 17

    [[17, 13], Detect_head,  [num_det_class, [[10,14,  23,27,  37,58], [81,82,  135,169,  344,319]], [256, 512]]], #Detection head 24

]


yolov4_tiny_mask_advance_1branch = [
    [18],   #Det_out_idx
    # backbone
    [-1, ConvBNReLU_mask, [3, 32, 3, 2, 1]],  # 0-P1/2
    [-1, ConvBNReLU_mask, [32, 64, 3, 2, 1]],  # 1-P2/4

    [-1, ConvBNReLU_mask, [64, 64, 3, 1, 1]],
    [-1, VoVCSP_mask, [64, 64]],
    # [[-1, 2], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 4-P3/8

    [-1, ConvBNReLU_mask, [128, 128, 3, 1, 1]],
    [-1, VoVCSP_mask, [128, 128]],
    # [[-1, 6], Concat, [1]],

    [-1, MaxPool2d, [2, 2]],  # 7-P4/16

    [-1, ConvBNReLU_mask, [256, 256, 3, 1, 1]],    # 8
    # [-1, VoV_mask, [256, 256]],
    # [[-1, 8], Concat, [1]],
    [-1, VoVCSP_mask, [256, 256]],      # 9

    [-1, MaxPool2d, [2, 2]],  # 10-P5/32

    [-1, ConvBNReLU_mask, [512, 512, 3, 1, 1]],  # 11
    # head
    [-1, ConvBNReLU_mask, [512, 256, 1, 1, 0]],
    [-1, ConvBNReLU_mask, [256, 512, 3, 1, 1]],   # 13

    [-1, ConvBNReLU_mask, [512, 256, 1, 1, 0]],    # 14
    [-1, ConvTranspose2d, [256, 256, 4, 2, 1, 0, 256, False]],  #
    # [-1, Upsample, [None, 2, 'nearest']],
    [[-1, 9], Concat, [1]],
    [-1, ConvBNReLU_mask, [512+256, 256, 3, 1, 1]],   # 17

    [[17], Detect_head,  [num_det_class, [[10, 14,  37, 58,  135, 169]], [256]]], #Detection head 24
    # [[10,14,  23,27,  37,58], [81,82,  135,169,  344,319]]

]



