import time
# time.sleep(2*60*60)

import warnings
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.ssd import SSD_Prune
from nets.ssd_loss import MultiboxLoss
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.dataloader import SSDDataset, ssd_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from nets.mobilenetv2 import MobileNetV2_4_2_prune


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    root = './logs/loss_2023_01_30_09_42_37_MobileNetV2_4_2/'
    name = '7.766888946817632'
    params_file = root+name+'.txt'
    weight_file = root+name+'.pth'
    Cuda = True
    classes_path    = 'data/voc_coner_point_classes.txt'
    # classes_path    = 'data/voc_byd_classes.txt'
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    train_annotation_path   = '2012_train_coner_point.txt'
    val_annotation_path     = '2012_val_coner_point.txt'
    model_path = ''
    #------------------------------------------------------#
    #   输入的shape大小
    #------------------------------------------------------#
    input_shape = [256, 256]
    #--------------------------------------------#
    # mobilenetv1_half, mobilenetv1, mobilenetv1_MFR, vgg, mobilenetv2, mobilenetv2_half，JacintoNetV2_lite,
    #---------------------------------------------#
    backbone = "MobileNetV2_4_2"
    pretrained = False
    batch_size = 32
    num_workers = 10
    start_epoch = 0
    end_epoch = 100
    lr = 1e-2  # initial learning rate (SGD=1E-2, Adam=1E-3)
    optimizer_type = 'sgd'
    lr_scheduler_type = 'cosine'    # cosine, cosine_wr

    num_batch = 0
    num_warmup = 0
    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, backbone)
    with open(params_file) as f:
        net_params = f.readlines()
    net_params = [[3, 24, (3, 3), (2, 2), (1, 1), 1], [[[24, 24, (3, 3), (1, 1), (1, 1), 24], [24, 16, (1, 1), (1, 1), (0, 0), 1]], 1, False], [[[16, 32, (1, 1), (1, 1), (0, 0), 1], [32, 32, (3, 3), (2, 2), (1, 1), 32], [32, 24, (1, 1), (1, 1), (0, 0), 1]], 4, False], [[[24, 40, (1, 1), (1, 1), (0, 0), 1], [40, 40, (3, 3), (1, 1), (1, 1), 40], [40, 24, (1, 1), (1, 1), (0, 0), 1]], 4, True], [[[24, 72, (1, 1), (1, 1), (0, 0), 1], [72, 72, (3, 3), (2, 2), (1, 1), 72], [72, 32, (1, 1), (1, 1), (0, 0), 1]], 4, False], [[[32, 72, (1, 1), (1, 1), (0, 0), 1], [72, 72, (3, 3), (1, 1), (1, 1), 72], [72, 32, (1, 1), (1, 1), (0, 0), 1]], 4, True], [[[32, 72, (1, 1), (1, 1), (0, 0), 1], [72, 72, (3, 3), (1, 1), (1, 1), 72], [72, 32, (1, 1), (1, 1), (0, 0), 1]], 4, True], [[[32, 88, (1, 1), (1, 1), (0, 0), 1], [88, 88, (3, 3), (2, 2), (1, 1), 88], [88, 64, (1, 1), (1, 1), (0, 0), 1]], 4, False], [[[64, 200, (1, 1), (1, 1), (0, 0), 1], [200, 200, (3, 3), (1, 1), (1, 1), 200], [200, 64, (1, 1), (1, 1), (0, 0), 1]], 4, True], [[[64, 152, (1, 1), (1, 1), (0, 0), 1], [152, 152, (3, 3), (1, 1), (1, 1), 152], [152, 64, (1, 1), (1, 1), (0, 0), 1]], 4, True], [[[64, 128, (1, 1), (1, 1), (0, 0), 1], [128, 128, (3, 3), (1, 1), (1, 1), 128], [128, 64, (1, 1), (1, 1), (0, 0), 1]], 4, True], [64, 24, (3, 3), (1, 1), (1, 1), 1], [24, 24, (4, 4), (2, 2), (1, 1), 0, 24], [], [[[56, 176, (1, 1), (1, 1), (0, 0), 1], [176, 176, (3, 3), (1, 1), (1, 1), 176], [176, 56, (1, 1), (1, 1), (0, 0), 1]], 2, False], [56, 16, (3, 3), (2, 2), (1, 1), 1], [], [[[80, 96, (1, 1), (1, 1), (0, 0), 1], [96, 96, (3, 3), (1, 1), (1, 1), 96], [96, 48, (1, 1), (1, 1), (0, 0), 1]], 2, False], [[[56, 64, (1, 1), (1, 1), (0, 0), 1], [64, 64, (3, 3), (1, 1), (1, 1), 64], [64, 104, (1, 1), (1, 1), (0, 0), 1]], 2, False], [104, 16, (3, 3), (1, 1), (1, 1), 1], [104, 24, (3, 3), (1, 1), (1, 1), 1], [[[48, 112, (1, 1), (1, 1), (0, 0), 1], [112, 112, (3, 3), (1, 1), (1, 1), 112], [112, 144, (1, 1), (1, 1), (0, 0), 1]], 2, False], [144, 16, (3, 3), (1, 1), (1, 1), 1], [144, 24, (3, 3), (1, 1), (1, 1), 1]]
    model = SSD_Prune(net_params, block_cfg=MobileNetV2_4_2_prune,
                      num_classes=num_classes)  # UNet_MobileNetV2_RM,UNet_MobileNetV2_4_RM
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['state_dict'])

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    criterion       = MultiboxLoss(num_classes, neg_pos_ratio=3.0)
    loss_history    = LossHistory("logs/", backbone+'_finetune')

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    train_dataset = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, augment=True)
    val_dataset = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, augment=False)

    gen_train = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model_train.parameters(), lr=lr, betas=(0.937, 0.999), weight_decay=1e-8)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model_train.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8, nesterov=True)
    else:
        optimizer = optim.SGD(model_train.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8, nesterov=True)

    if lr_scheduler_type == 'cosine_wr':
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    elif lr_scheduler_type == 'cosine':
        num_batch = len(gen_train)
        lf = lambda x: ((1 + math.cos(x * math.pi / end_epoch)) / 2) * (1 - 0.2) + 0.2  # cosine
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        num_warmup = max(round(3.0 * num_batch), 1000)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)


    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, criterion, loss_history, optimizer, epoch,
                epoch_step, epoch_step_val, gen_train, gen_val, end_epoch, Cuda,
                      num_batch=num_batch, num_warmup=num_warmup, END_EPOCH=end_epoch, sparsity=0)

        lr_scheduler.step()


