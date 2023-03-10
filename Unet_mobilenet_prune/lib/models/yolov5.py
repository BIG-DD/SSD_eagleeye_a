import sys,os
sys.path.append(os.getcwd())
from lib.models.common import ConvBNReLU, SPP, BottleneckCSP, Focus, Concat, Detect, Detect_head, Segment_head, C3, InvertedResidual
from torch.nn import Upsample, ConvTranspose2d
from lib.config import cfg

num_det_class = cfg.num_det_class

# n_ = max(round(n * gd=0.33), 1) if n > 1 else n  # depth gain
#  c_ = math.ceil(c * gw / 8) * 8 # channels gain # mobilenet only modify channels
# The lane line and the driving area segment branches without share information with each other and without link
YOLOP_3head = [
    [24, 33, 42],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],   #0
    [-1, ConvBNReLU, [32, 64, 3, 2]],    #1
    [-1, BottleneckCSP, [64, 64, 1]],  #2
    [-1, ConvBNReLU, [64, 128, 3, 2]],   #3
    [-1, BottleneckCSP, [128, 128, 3]],    #4
    [-1, ConvBNReLU, [128, 256, 3, 2]],  #5
    [-1, BottleneckCSP, [256, 256, 3]],    #6
    [-1, ConvBNReLU, [256, 512, 3, 2]],  #7
    [-1, SPP, [512, 512, [5, 9, 13]]],     #8
    [-1, BottleneckCSP, [512, 512, 1, False]],     #9
    [-1, ConvBNReLU,[512, 256, 1, 1]],   #10
    [-1, Upsample, [None, 2, 'nearest']],  #11
    [[-1, 6], Concat, [1]],    #12
    [-1, BottleneckCSP, [512, 256, 1, False]], #13
    [-1, ConvBNReLU, [256, 128, 1, 1]],  #14
    [-1, Upsample, [None, 2, 'nearest']],  #15
    [[-1,4], Concat, [1]],     #16         #Encoder

    [-1, BottleneckCSP, [256, 128, 1, False]],     #17
    [-1, ConvBNReLU, [128, 128, 3, 2]],      #18
    [[-1, 14], Concat, [1]],       #19
    [-1, BottleneckCSP, [256, 256, 1, False]],     #20
    [-1, ConvBNReLU, [256, 256, 3, 2]],      #21
    [[-1, 10], Concat, [1]],   #22
    [-1, BottleneckCSP, [512, 512, 1, False]],     #23
    [[17, 20, 23], Detect,  [num_det_class, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24

    [16, ConvBNReLU, [256, 128, 3, 1]],   #25
    [-1, Upsample, [None, 2, 'nearest']],  #26
    [-1, BottleneckCSP, [128, 64, 1, False]],  #27
    [-1, ConvBNReLU, [64, 32, 3, 1]],    #28
    [-1, Upsample, [None, 2, 'nearest']],  #29
    [-1, ConvBNReLU, [32, 16, 3, 1]],    #30
    [-1, BottleneckCSP, [16, 8, 1, False]],    #31
    [-1, Upsample, [None, 2, 'nearest']],  #32
    [-1, ConvBNReLU, [8, 2, 3, 1]], #33 Driving area segmentation head

    [16, ConvBNReLU, [256, 128, 3, 1]],   #34
    [-1, Upsample, [None, 2, 'nearest']],  #35
    [-1, BottleneckCSP, [128, 64, 1, False]],  #36
    [-1, ConvBNReLU, [64, 32, 3, 1]],    #37
    [-1, Upsample, [None, 2, 'nearest']],  #38
    [-1, ConvBNReLU, [32, 16, 3, 1]],    #39
    [-1, BottleneckCSP, [16, 8, 1, False]],    #40
    [-1, Upsample, [None, 2, 'nearest']],  #41
    [-1, ConvBNReLU, [8, 2, 3, 1]] #42 Lane line segmentation head
]


YOLOP_2head = [
    [24, 33],   #Det_out_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],   # 0-P1/2 384->192
    [-1, ConvBNReLU, [32, 64, 3, 2]],    # 1-P2/4 192->96
    [-1, BottleneckCSP, [64, 64, 1]],  # 2
    [-1, ConvBNReLU, [64, 128, 3, 2]],     # 3-P3/8 96->48
    [-1, BottleneckCSP, [128, 128, 3]],    # 4
    [-1, ConvBNReLU, [128, 256, 3, 2]],  # 5-P3/16 48->24
    [-1, BottleneckCSP, [256, 256, 3]],    # 6
    [-1, ConvBNReLU, [256, 512, 3, 2]],  # 7-P3/32 24->12
    [-1, SPP, [512, 512, [5, 9, 13]]],     # 8
    [-1, BottleneckCSP, [512, 512, 1, False]],     # 9

    [-1, ConvBNReLU, [512, 256, 1, 1]],   # 10
    [-1, Upsample, [None, 2, 'nearest']],  # 11-P3/16 12->24
    [[-1, 6], Concat, [1]],    # 12
    [-1, BottleneckCSP, [512, 256, 1, False]],  # 13
    [-1, ConvBNReLU, [256, 128, 1, 1]],  # 14
    [-1, Upsample, [None, 2, 'nearest']],  # 15-P3/8 24->48
    [[-1, 4], Concat, [1]],     # 16         #Encoder

    [-1, BottleneckCSP, [256, 128, 1, False]],     # 17

    [-1, ConvBNReLU, [128, 128, 3, 2]],      # 18-P3/16 48->24
    [[-1, 14], Concat, [1]],       # 19
    [-1, BottleneckCSP, [256, 256, 1, False]],     # 20

    [-1, ConvBNReLU, [256, 256, 3, 2]],      # 21-P3/32 24->12
    [[-1, 10], Concat, [1]],   # 22
    [-1, BottleneckCSP, [512, 512, 1, False]],     # 23

    [[17, 20, 23], Detect_head,  [num_det_class, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]], #Detection head 24

    # [ 16, Conv, [256, 128, 3, 1]],   #25
    # [ -1, Upsample, [None, 2, 'nearest']],  #26
    # [ -1, BottleneckCSP, [128, 64, 1, False]],  #27
    # [ -1, Conv, [64, 32, 3, 1]],    #28
    # [ -1, Upsample, [None, 2, 'nearest']],  #29
    # [ -1, Conv, [32, 16, 3, 1]],    #30
    # [ -1, BottleneckCSP, [16, 8, 1, False]],    #31
    # [ -1, Upsample, [None, 2, 'nearest']],  #32
    # [ -1, Conv, [8, 2, 3, 1]], #33 Driving area segmentation head

    [16, ConvBNReLU, [256, 128, 3, 1]],   #25
    [-1, Upsample, [None, 2, 'nearest']],  #26
    [-1, BottleneckCSP, [128, 64, 1, False]],  #27
    [-1, ConvBNReLU, [64, 32, 3, 1]],    #28
    [-1, Upsample, [None, 2, 'nearest']],  #29
    [-1, ConvBNReLU, [32, 16, 3, 1]],    #30
    [-1, BottleneckCSP, [16, 8, 1, False]],    #31
    [-1, Upsample, [None, 2, 'nearest']],  #32
    # [-1, ConvBNReLU, [8, 2, 3, 1]] #33 Lane line segmentation head
    [32, Segment_head, [8, 2, 3, 1]]  # 33 segmentation head
]


YOLOV5l6_2head = [
    [30, 39],   #Det_out_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],  # 0-P1/2 384->192
    [-1, ConvBNReLU, [32, 64, 3, 2]],  # 1-P2/4 192->96
    [-1, C3, [64, 64, 1]],
    [-1, ConvBNReLU, [64, 128, 3, 2]],  # 3-P3/8 96->48
    [-1, C3, [128, 128, 3]],
    [-1, ConvBNReLU, [128, 256, 3, 2]],  # 5-P4/16 48->24
    [-1, C3, [256, 256, 3]],
    [-1, ConvBNReLU, [256, 384, 3, 2]],  # 7-P5/32 24->12
    [-1, C3, [384, 384, 1]],
    [-1, ConvBNReLU, [384, 512, 3, 2]],  # 9-P6/64 12->6
    [-1, SPP, [512, 512, [3, 5, 7]]],   # SPP, [512, 512, [5, 9, 13]]
    [-1, C3, [512, 512, 1, False]],  # 11

    [-1, ConvBNReLU, [512, 384, 1, 1]],
    [-1, Upsample, [None, 2, 'nearest']],    # 6->12
    [[-1, 8], Concat, [1]],  # cat backbone P5
    [-1, C3, [768, 384, 1, False]],  # 15

    [-1, ConvBNReLU, [384, 256, 1, 1]],
    [-1, Upsample, [None, 2, 'nearest']],    # 12->24
    [[-1, 6], Concat, [1]],  # cat backbone P4
    [-1, C3, [512, 256, False]],  # 19

    [-1, ConvBNReLU, [256, 128, 1, 1]],
    [-1, Upsample, [None, 2, 'nearest']],    # 24->48
    [[-1, 4], Concat, [1]],  # cat backbone P3
    [-1, C3, [256, 128, False]],  # 23 (P3/8-small)

    [-1, ConvBNReLU, [128, 128, 3, 2]],    # 48->24
    [[-1, 20], Concat, [1]],  # cat head P4
    [-1, C3, [256, 256, 1, False]],  # 26 (P4/16-medium)

    [-1, ConvBNReLU, [256, 256, 3, 2]],    # 24->12
    [[-1, 16], Concat, [1]],  # cat head P5
    [-1, C3, [512, 512, 1, False]],  # 29 (P5/32-large)

    # [-1, Conv, [384, 384, 3, 2]],    # 12->6
    # [[-1, 12], Concat, [1]],  # cat head P6
    # [-1, C3, [768, 512, 1, False]],  # 32 (P6/64-xlarge)

    [[23, 26, 29], Detect_head, [num_det_class, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],  # 30 Detect(P3, P4, P5, P6)

    [22, ConvBNReLU, [256, 128, 3, 1]],   #
    [-1, Upsample, [None, 2, 'nearest']],  #     # 48->96
    [-1, C3, [128, 64, 1, False]],  #
    [-1, ConvBNReLU, [64, 32, 3, 1]],    #
    [-1, Upsample, [None, 2, 'nearest']],  #    # 96->192
    [-1, ConvBNReLU, [32, 16, 3, 1]],    #
    [-1, C3, [16, 8, 1, False]],    #
    [-1, Upsample, [None, 2, 'nearest']],  #     # 192->384
    # [-1, ConvBNReLU, [8, 2, 3, 1]]  # 39 Lane line segmentation head
    [-1, Segment_head, [8, 2, 3, 1]]  # 33 segmentation head
]


YOLOV5S_2head = [
    [24, 33],  # Det_out_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],  # 0-P1/2 384->192
    [-1, ConvBNReLU, [32, 64, 3, 2, 1]],  # 1-P2/4 192->96
    [-1, C3, [64, 64, 1]],  # 2
    [-1, ConvBNReLU, [64, 128, 3, 2, 1]],  # 3-P3/8 96->48
    [-1, C3, [128, 128, 3]],  # 4
    [-1, ConvBNReLU, [128, 256, 3, 2, 1]],  # 5-P3/16 48->24
    [-1, C3, [256, 256, 3]],  # 6
    [-1, ConvBNReLU, [256, 512, 3, 2, 1]],  # 7-P3/32 24->12
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 8
    [-1, C3, [512, 512, 1, False]],  # 9

    [-1, ConvBNReLU, [512, 256, 1, 1, 0]],  # 10
    [-1, Upsample, [None, 2, 'nearest']],  # 11-P3/16 12->24
    [[-1, 6], Concat, [1]],  # 12
    [-1, C3, [512, 256, 1, False]],  # 13

    [-1, ConvBNReLU, [256, 128, 1, 1, 0]],  # 14
    [-1, Upsample, [None, 2, 'nearest']],  # 15-P3/8 24->48
    [[-1, 4], Concat, [1]],  # 16         #Encoder
    [-1, C3, [256, 128, 1, False]],  # 17

    [-1, ConvBNReLU, [128, 128, 3, 2, 1]],  # 18-P3/16 48->24
    [[-1, 14], Concat, [1]],  # 19
    [-1, C3, [256, 256, 1, False]],  # 20

    [-1, ConvBNReLU, [256, 256, 3, 2, 1]],  # 21-P3/32 24->12
    [[-1, 10], Concat, [1]],  # 22
    [-1, C3, [512, 512, 1, False]],  # 23

    [[17, 20, 23], Detect_head,
     [num_det_class, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],    # Detection head 24

    [16, ConvBNReLU, [256, 128, 3, 1, 1]],  # 25
    [-1, Upsample, [None, 2, 'nearest']],  # 26
    [-1, C3, [128, 64, 1, False]],  # 27
    [-1, ConvBNReLU, [64, 32, 3, 1, 1]],  # 28
    [-1, Upsample, [None, 2, 'nearest']],  # 29
    [-1, ConvBNReLU, [32, 16, 3, 1, 1]],  # 30
    [-1, C3, [16, 8, 1, False]],  # 31
    [-1, Upsample, [None, 2, 'nearest']],  # 32
    # [-1, ConvBNReLU, [8, 2, 3, 1]]  # 33 segmentation head
    [-1, Segment_head, [8, 2, 3, 1, 1]]  # 33 segmentation head
  ]


YOLOV5S_segment = [
    [25],  # Det_out_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],  # 0-P1/2 384->192#256->128
    [-1, ConvBNReLU, [32, 64, 3, 2, 1]],  # 1-P2/4 192->96#128->64///inp, oup, kernel_size=3, stride=1, padding=0
    [-1, C3, [64, 64, 1]],  # 2
    [-1, ConvBNReLU, [64, 128, 3, 2, 1]],  # 3-P3/8 96->48#64->32
    [-1, C3, [128, 128, 3]],  # 4
    [-1, ConvBNReLU, [128, 256, 3, 2, 1]],  # 5-P3/16 48->24#32->16
    [-1, C3, [256, 256, 3]],  # 6
    [-1, ConvBNReLU, [256, 512, 3, 2, 1]],  # 7-P3/32 24->12#16->8
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 8
    [-1, C3, [512, 512, 1, False]],  # 9

    [-1, ConvBNReLU, [512, 256, 1, 1, 0]],  # 10
    [-1, Upsample, [None, 2, 'nearest']],  # 11-P3/16 12->24#8->16
    [[-1, 6], Concat, [1]],  # 12
    [-1, C3, [512, 256, 1, False]],  # 13

    [-1, ConvBNReLU, [256, 128, 1, 1, 0]],  # 14
    [-1, Upsample, [None, 2, 'nearest']],  # 15-P3/8 24->48#16->32
    [[-1, 4], Concat, [1]],  # 16         #Encoder

    [16, ConvBNReLU, [256, 128, 3, 1, 1]],  # 17
    [-1, Upsample, [None, 2, 'nearest']],  # 18#32->64
    [-1, C3, [128, 64, 1, False]],  # 19
    [-1, ConvBNReLU, [64, 32, 3, 1, 1]],  # 20
    [-1, Upsample, [None, 2, 'nearest']],  # 21#64->128
    [-1, ConvBNReLU, [32, 16, 3, 1, 1]],  # 22
    [-1, C3, [16, 8, 1, False]],  # 23
    [-1, Upsample, [None, 2, 'nearest']],  # 24#128->256
    # [-1, ConvBNReLU, [8, 2, 3, 1]]  # 33 segmentation head
    [-1, Segment_head, [8, 2, 3, 1, 1]]  # 25 segmentation head
  ]

