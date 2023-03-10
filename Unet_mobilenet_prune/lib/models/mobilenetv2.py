import sys,os
sys.path.append(os.getcwd())
from lib.models.common import ConvBNReLU, Concat, Detect_head, Segment_head, C3, InvertedResidual
from torch.nn import Upsample, ConvTranspose2d
from lib.config import cfg

num_det_class = cfg.num_det_class

# mobilenetv2
# t, c, n, s
# [1, 16, 1, 1],
# [6, 24, 2, 2],
# [6, 32, 3, 2],
# [6, 64, 4, 2],
# [6, 96, 3, 1],
# [6, 160, 3, 2],
# [6, 320, 1, 1],

# mobilenetV2_1
mobilenetV2_1_2head = [
    [32, 41],  # Det_out_idx, LL_Segout_idx
# backbone:
    [-1, ConvBNReLU, [3, 32, 3, 2]],              # 0-P1/2   320x320*32
    [-1, InvertedResidual, [32, 16, 1, 1]],       # 1        320x320*16
    [-1, InvertedResidual, [16, 24, 2, 6]],       # 2-P2/4   160x160*24
    [-1, InvertedResidual, [24, 24, 1, 6]],       # 3-P2/4   160x160*24
    [-1, InvertedResidual, [24, 32, 2, 6]],       # 4-P3/8   80x80*32
    [-1, InvertedResidual, [32, 32, 1, 6]],       # 5-P3/8   80x80*32
    [-1, InvertedResidual, [32, 32, 1, 6]],       # 6-P3/8   80x80*32
    [-1, InvertedResidual, [32, 64, 2, 6]],       # 7-P4/16  40x40*64
    [-1, InvertedResidual, [64, 64, 1, 6]],       # 8-P4/16  40x40*64
    [-1, InvertedResidual, [64, 64, 1, 6]],       # 9-P4/16  40x40*64
    [-1, InvertedResidual, [64, 64, 1, 6]],       # 10-P4/16 40x40*64
    [-1, InvertedResidual, [64, 96, 1, 6]],       # 11       40X40*96
    [-1, InvertedResidual, [96, 96, 1, 6]],       # 12       40X40*96
    [-1, InvertedResidual, [96, 96, 1, 6]],       # 13       40X40*96
    [-1, InvertedResidual, [96, 160, 2, 6]],      # 14-P5/32  20X20*160
    [-1, InvertedResidual, [160, 160, 1, 6]],     # 15-P5/32  20X20*160
    [-1, InvertedResidual, [160, 160, 1, 6]],     # 16-P5/32  20X20*160
    [-1, InvertedResidual, [160, 320, 1, 6]],     # 17       20X20*320
# neck:
    [-1, ConvBNReLU, [320, 256, 1, 1]],           # 18       20*20*320  -->  20*20*512
    # [-1, Upsample, [None, 2, 'nearest']],         # 19       20*20*512  -->  40*40*512
    [-1, ConvTranspose2d, [256, 256, 4, 2, 1, 0, 256, False]],
    [[-1, 10], Concat, [1]],  # cat backbone P4   # 20       40*40*512  +    40*40*64  -->  40*40*576
    [-1, C3, [320, 256, False]],  # 13            # 21       head-P4         #    40*40*576  -->  40*40*512

    [-1, ConvBNReLU, [256, 128, 1, 1]],           # 22       40*40*512  -->  40*40*256
    # [-1, Upsample, [None, 2, 'nearest']],         # 23       40*40*256  -->  80*80*256
    [-1, ConvTranspose2d, [128, 128, 4, 2, 1, 0, 128, False]],
    [[-1, 6], Concat, [1]],  # cat backbone P3    # 24       80*80*256  +    80*80*32  -->  80*80*288
# head Detect:
    [-1, C3, [160, 128, False]],  # 17            # 25 (P3/8-small)      #    80*80*288  -->  80*80*256   25

    [-1, ConvBNReLU, [128, 128, 3, 2]],           # 26       80*80*256  -->  40*40*256
    [[-1, 13], Concat, [1]],  # cat head P4       # 27       40*40*256  +    40*40*512   -->  40*40*768
    [-1, C3, [224, 256, False]],  # 20            # 28  (P4/16-medium)    #    40*40*768  -->  40*40*512   28

    [-1, ConvBNReLU, [256, 256, 3, 2]],           # 29       40*40*512  -->  20*20*512
    [[-1, 18], Concat, [1]],  # cat head P5       # 30       20*20*512  +    20X20*320   -->  20*20*832
    [-1, C3, [512, 512, False]],  # 23            # 31 (P5/32-large)    #    20*20*832  -->  20*20*1024   31

    # [[25, 28, 31], 1, Detect, [nc, anchors]],   # Detect(P3, P4, P5)
    [[25, 28, 31], Detect_head,
     [num_det_class, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],
# head segmentation:
    [24, ConvBNReLU, [160, 128, 3, 1]],           # 33
    # [-1, Upsample, [None, 2, 'nearest']],         # 34
    [-1, ConvTranspose2d, [128, 128, 4, 2, 1, 0, 128, False]],
    [-1, C3, [128, 64, 1, False]],                # 35
    [-1, ConvBNReLU, [64, 32, 3, 1]],             # 36
    # [-1, Upsample, [None, 2, 'nearest']],         # 37
    [-1, ConvTranspose2d, [32, 32, 4, 2, 1, 0, 32, False]],
    [-1, ConvBNReLU, [32, 16, 3, 1]],             # 38
    [-1, C3, [16, 8, 1, False]],                  # 39
    # [-1, Upsample, [None, 2, 'nearest']],         # 40
    [-1, ConvTranspose2d, [8, 8, 4, 2, 1, 0, 8, False]],
    [-1, Segment_head, [8, 2, 3, 1]]              # 41 segmentation head

  ]

mobilenetV2_half_2head = [
    [32, 41],  # Det_out_idx, LL_Segout_idx
# backbone:
    [-1, ConvBNReLU, [3, 16, 3, 2]],              # 0-P1/2   320x320*32
    [-1, InvertedResidual, [16, 8, 1, 1]],        # 1        320x320*16
    [-1, InvertedResidual, [8, 12, 2, 6]],        # 2-P2/4   160x160*24
    [-1, InvertedResidual, [12, 12, 1, 6]],       # 3-P2/4   160x160*24
    [-1, InvertedResidual, [12, 16, 2, 6]],       # 4-P3/8   80x80*32
    [-1, InvertedResidual, [16, 16, 1, 6]],       # 5-P3/8   80x80*32
    [-1, InvertedResidual, [16, 16, 1, 6]],       # 6-P3/8   80x80*32
    [-1, InvertedResidual, [16, 32, 2, 6]],       # 7-P4/16  40x40*64
    [-1, InvertedResidual, [32, 32, 1, 6]],       # 8-P4/16  40x40*64
    [-1, InvertedResidual, [32, 32, 1, 6]],       # 9-P4/16  40x40*64
    [-1, InvertedResidual, [32, 32, 1, 6]],       # 10-P4/16 40x40*64
    [-1, InvertedResidual, [32, 48, 1, 6]],       # 11       40X40*96
    [-1, InvertedResidual, [48, 48, 1, 6]],       # 12       40X40*96
    [-1, InvertedResidual, [48, 48, 1, 6]],       # 13       40X40*96
    [-1, InvertedResidual, [48, 80, 2, 6]],       # 14-P5/32  20X20*160
    [-1, InvertedResidual, [80, 80, 1, 6]],       # 15-P5/32  20X20*160
    [-1, InvertedResidual, [80, 80, 1, 6]],       # 16-P5/32  20X20*160
    [-1, InvertedResidual, [80, 160, 1, 6]],      # 17       20X20*320
# neck:
    [-1, ConvBNReLU, [160, 256, 1, 1]],           # 18       20*20*320  -->  20*20*512
    [-1, ConvTranspose2d, [256, 256, 4, 2, 1, 0, 256, False]],      # 19       20*20*512  -->  40*40*512
    [[-1, 10], Concat, [1]],  # cat backbone P4   # 20       40*40*512  +    40*40*64  -->  40*40*576
    [-1, C3, [256+32, 256, False]],  # 13            # 21       head-P4         #    40*40*576  -->  40*40*512

    [-1, ConvBNReLU, [256, 128, 1, 1]],           # 22       40*40*512  -->  40*40*256
    [-1, ConvTranspose2d, [128, 128, 4, 2, 1, 0, 128, False]],   # 23       40*40*256  -->  80*80*256
    [[-1, 6], Concat, [1]],  # cat backbone P3    # 24       80*80*256  +    80*80*32  -->  80*80*288
# head Detect:
    [-1, C3, [128+16, 128, False]],  # 17            # 25 (P3/8-small)      #    80*80*288  -->  80*80*256   25

    [-1, ConvBNReLU, [128, 128, 3, 2]],           # 26       80*80*256  -->  40*40*256
    [[-1, 13], Concat, [1]],  # cat head P4       # 27       40*40*256  +    40*40*512   -->  40*40*768
    [-1, C3, [128+48, 256, False]],  # 20            # 28  (P4/16-medium)    #    40*40*768  -->  40*40*512   28

    [-1, ConvBNReLU, [256, 256, 3, 2]],           # 29       40*40*512  -->  20*20*512
    [[-1, 18], Concat, [1]],  # cat head P5       # 30       20*20*512  +    20X20*320   -->  20*20*832
    [-1, C3, [256+256, 512, False]],  # 23            # 31 (P5/32-large)    #    20*20*832  -->  20*20*1024   31

    # [[25, 28, 31], 1, Detect, [nc, anchors]],   # Detect(P3, P4, P5)
    [[25, 28, 31], Detect_head,
     [num_det_class, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],
# head segmentation:
    [24, ConvBNReLU, [128+16, 128, 3, 1]],           # 33
    [-1, ConvTranspose2d, [128, 128, 4, 2, 1, 0, 128, False]],         # 34
    [-1, C3, [128, 64, 1, False]],                # 35
    [-1, ConvBNReLU, [64, 32, 3, 1]],             # 36
    [-1, ConvTranspose2d, [32, 32, 4, 2, 1, 0, 32, False]],         # 37
    [-1, ConvBNReLU, [32, 16, 3, 1]],             # 38
    [-1, C3, [16, 8, 1, False]],                  # 39
    [-1, ConvTranspose2d, [8, 8, 4, 2, 1, 0, 8, False]],         # 40
    [-1, Segment_head, [8, 2, 3, 1]]              # 41 segmentation head

  ]

# mobilenetV2_0.25 modify channels
mobilenetV2_quarter_2head = [
    [32, 41],  # Det_out_idx, LL_Segout_idx
# backbone:
    [-1, ConvBNReLU, [3, 8, 3, 2]],               # 0-P1/2   320x320*32
    [-1, InvertedResidual, [8, 8, 1, 1]],         # 1        320x320*16
    [-1, InvertedResidual, [8, 8, 2, 6]],         # 2-P2/4   160x160*24
    [-1, InvertedResidual, [8, 8, 1, 6]],         # 3-P2/4   160x160*24
    [-1, InvertedResidual, [8, 8, 2, 6]],         # 4-P3/8   80x80*32
    [-1, InvertedResidual, [8, 8, 1, 6]],         # 5-P3/8   80x80*32
    [-1, InvertedResidual, [8, 8, 1, 6]],         # 6-P3/8   80x80*32
    [-1, InvertedResidual, [8, 16, 2, 6]],        # 7-P4/16  40x40*64
    [-1, InvertedResidual, [16, 16, 1, 6]],       # 8-P4/16  40x40*64
    [-1, InvertedResidual, [16, 16, 1, 6]],       # 9-P4/16  40x40*64
    [-1, InvertedResidual, [16, 16, 1, 6]],       # 10-P4/16 40x40*64
    [-1, InvertedResidual, [16, 24, 1, 6]],       # 11       40X40*96
    [-1, InvertedResidual, [24, 24, 1, 6]],       # 12       40X40*96
    [-1, InvertedResidual, [24, 24, 1, 6]],       # 13       40X40*96
    [-1, InvertedResidual, [24, 40, 2, 6]],       # 14-P5/32  20X20*160
    [-1, InvertedResidual, [40, 40, 1, 6]],       # 15-P5/32  20X20*160
    [-1, InvertedResidual, [40, 40, 1, 6]],       # 16-P5/32  20X20*160
    [-1, InvertedResidual, [40, 80, 1, 6]],       # 17       20X20*320
# neck:
    [-1, ConvBNReLU, [80, 256, 1, 1]],            # 18       20*20*320  -->  20*20*512
    # [-1, Upsample, [None, 2, 'nearest']],      # 19       20*20*512  -->  40*40*512
    [-1, ConvTranspose2d, [256, 256, 4, 2, 1, 0, 256, False]],  # 19       20*20*512  -->  40*40*512
    [[-1, 10], Concat, [1]],  # cat backbone P4   # 20       40*40*512  +    40*40*64  -->  40*40*576
    [-1, C3, [272, 256, False]],  # 13            # 21       head-P4         #    40*40*576  -->  40*40*512

    [-1, ConvBNReLU, [256, 128, 1, 1]],           # 22       40*40*512  -->  40*40*256
    # [-1, Upsample, [None, 2, 'nearest']],   # 23       40*40*256  -->  80*80*256
    [-1, ConvTranspose2d, [128, 128, 4, 2, 1, 0, 128, False]],   # 23       40*40*256  -->  80*80*256
    [[-1, 6], Concat, [1]],  # cat backbone P3    # 24       80*80*256  +    80*80*32  -->  80*80*288
# head Detect:
    [-1, C3, [136, 128, False]],  # 17            # 25 (P3/8-small)      #    80*80*288  -->  80*80*256   25

    [-1, ConvBNReLU, [128, 128, 3, 2]],           # 26       80*80*256  -->  40*40*256
    [[-1, 13], Concat, [1]],  # cat head P4       # 27       40*40*256  +    40*40*512   -->  40*40*768
    [-1, C3, [152, 256, False]],  # 20            # 28  (P4/16-medium)    #    40*40*768  -->  40*40*512   28

    [-1, ConvBNReLU, [256, 256, 3, 2]],           # 29       40*40*512  -->  20*20*512
    [[-1, 18], Concat, [1]],  # cat head P5       # 30       20*20*512  +    20X20*320   -->  20*20*832
    [-1, C3, [512, 512, False]],  # 23            # 31 (P5/32-large)    #    20*20*832  -->  20*20*1024   31

    [[25, 28, 31], Detect_head,
     [num_det_class, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],
# head segmentation:
    [24, ConvBNReLU, [136, 128, 3, 1]],           # 33
    # [-1, Upsample, [None, 2, 'nearest']],         # 34
    [-1, ConvTranspose2d, [128, 128, 4, 2, 1, 0, 128, False]],         # 34

    [-1, C3, [128, 64, 1, False]],                # 35
    [-1, ConvBNReLU, [64, 32, 3, 1]],             # 36
    # [-1, Upsample, [None, 2, 'nearest']],         # 37
    [-1, ConvTranspose2d, [32, 32, 4, 2, 1, 0, 32, False]],         # 37
    [-1, ConvBNReLU, [32, 16, 3, 1]],             # 38
    [-1, C3, [16, 8, 1, False]],                  # 39
    # [-1, Upsample, [None, 2, 'nearest']],         # 40
    [-1, ConvTranspose2d, [8, 8, 4, 2, 1, 0, 8, False]],         # 40
    [-1, Segment_head, [8, 2, 3, 1]]              # 41 segmentation head

  ]

