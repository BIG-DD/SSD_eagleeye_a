import os
from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = '/media/z590/F/Pytorch/YOLOP/tools/runs/'
_C.GPUS = (0)
_C.WORKERS = 10
_C.PIN_MEMORY = False
_C.PRINT_FREQ = 1000      # logger
_C.AUTO_RESUME = False       # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
_C.num_seg_class = 2
_C.num_det_class = 4

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
# YOLOV5S_2head, YOLOV5l6_2head, YOLOP_2head, YOLOP, YOLOV5S_mobilenetV2_half_2head, YOLOV5S_mobilenetV2_quarter_2head, YOLOV5S_mobilenetV2_1_2head
_C.MODEL.NAME = 'YOLOV5S_2head'
_C.MODEL.STRU_WITHSHARE = False     #add share_block to segbranch
_C.MODEL.HEADS_NAME = ['']
_C.MODEL.PRETRAINED = ''# "/media/z590/G/Pytorch/YOLOP-main/tools/runs/BddDataset/YOLOV5S_2head/_2021-12-04-09-52_train/best-epoch-237.pth"
_C.MODEL.FINETUNE = ''  # 减枝微调训练
_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.IMAGE_SIZE = [384, 384]  # height,width ex: 192, 256
_C.MODEL.EXTRA = CN(new_allowed=True)

# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 0.05  # box loss gain # 0.05
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
_C.LOSS.LL_SEG_GAIN = 0.2  # lane line segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.2  # lane line iou loss gain


# DATASET related params
_C.DATASET = CN(new_allowed=True)
root = '/media/z590/D/PublicDataSet/context-based-parking-slot-detect/YOLOP/caiji_train_dataset/'#'/media/z590/D/PublicDataSet/bdd100k/'#
_C.DATASET.DATAROOT = root+'images'       # the path of images folder
_C.DATASET.LABELROOT = root+'det_annotations'      # the path of det_annotations folder
_C.DATASET.MASKROOT = root+'da_seg_annotations'                # the path of da_seg_annotations folder
_C.DATASET.LANEROOT = root+'ll_seg_annotations'               # the path of ll_seg_annotations folder
_C.DATASET.DATASET = 'BddDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [768, 768]#[720, 1280]   #

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # images HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # images HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # images HSV-Value augmentation (fraction)
# TODO: more augmet params to add


# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.LRF = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 240

_C.TRAIN.VAL_FREQ = 10
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

_C.TRAIN.SPARSITY_FACTOR = 0  # 稀疏化因子
_C.TRAIN.PRUNE_SCALE_RATE = 0  # 裁减比例 prune_scale_rate

# if training 3 tasks end-to-end, set all parameters as False
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = True       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task

_C.TRAIN.PLOT = True                # 

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True
_C.TEST.NMS_CONF_THRESHOLD = 0.25    # 0.001
_C.TEST.NMS_IOU_THRESHOLD = 0.6


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.sparsity_factor:
        cfg.TRAIN.SPARSITY_FACTOR = args.sparsity_factor

    if args.finetune:
        cfg.MODEL.FINETUNE = args.finetune

    if args.prune_scale_rate:
        cfg.TRAIN.PRUNE_SCALE_RATE = args.prune_scale_rate





    # if args.conf_thres:
    #     cfg.TEST.NMS_CONF_THRESHOLD = args.conf_thres

    # if args.iou_thres:
    #     cfg.TEST.NMS_IOU_THRESHOLD = args.iou_thres
    


    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )
    #
    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()
