import time
# time.sleep(2*60*60)

import warnings
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.ssd import SSD300, SSD
from nets.ssd_loss import MultiboxLoss, weights_init
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.dataloader import SSDDataset, ssd_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.utils_bbox import BBoxUtility


warnings.filterwarnings("ignore")
'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
   这些都是经验上，只能靠各位同学多查询资料和自己试试了。
'''  
if __name__ == "__main__":
    Cuda = True
    classes_path    = 'data/voc_obstacle_classes.txt'
    # classes_path    = 'data/voc_byd_classes.txt'
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    train_annotation_path   = '2012_train.txt'
    val_annotation_path     = '2012_val.txt'
    model_path = ''
    #------------------------------------------------------#
    #   输入的shape大小
    #------------------------------------------------------#
    input_shape = [512, 512]
    #--------------------------------------------#
    # mobilenetv1_half, mobilenetv1, mobilenetv1_MFR, mobilenetv1_F_SSD, vgg, mobilenetv2, mobilenetv2_half，JacintoNetV2_lite,
    #---------------------------------------------#
    backbone = "mobilenetv1_F_SSD"
    pretrained = False
    batch_size = 32
    num_workers = 16
    start_epoch = 0
    end_epoch = 150
    lr = 1e-3  # initial learning rate (SGD=1E-2, Adam=1E-3)
    optimizer_type = 'adam'
    lr_scheduler_type = 'cosine'    # cosine, cosine_wr

    num_batch = 0
    num_warmup = 0
    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, backbone)

    # model = SSD300(num_classes, backbone, pretrained)
    backbone = 'MobileNetV2_4_2_test'
    model = SSD(backbone=backbone, num_classes=num_classes)

    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    criterion       = MultiboxLoss(num_classes, neg_pos_ratio=3.0)
    bbox_util       = BBoxUtility(class_names, num_classes, input_shape, input_shape, anchors, Cuda)
    loss_history    = LossHistory("logs/", backbone)

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
        fit_one_epoch(model_train, model, criterion, bbox_util, loss_history, optimizer, epoch,
                epoch_step, epoch_step_val, gen_train, gen_val, end_epoch, Cuda,
                      num_batch=num_batch, num_warmup=num_warmup, END_EPOCH=end_epoch, sparsity=0)

        lr_scheduler.step()

    # if True:
    #     batch_size  = Unfreeze_batch_size
    #     lr          = Unfreeze_lr
    #     start_epoch = Freeze_Epoch
    #     end_epoch   = UnFreeze_Epoch
    #
    #     epoch_step      = num_train // batch_size
    #     epoch_step_val  = num_val // batch_size
    #
    #     if epoch_step == 0 or epoch_step_val == 0:
    #         raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
    #     # optimizer = torch.optim.SGD(model_train.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #     optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    #     lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    #
    #     train_dataset   = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, augment=True)
    #     val_dataset     = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, augment=False)
    #
    #     gen             = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                                 drop_last=True, collate_fn=ssd_dataset_collate)
    #     gen_val         = DataLoader(val_dataset  , shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                                 drop_last=True, collate_fn=ssd_dataset_collate)
    #
    #     #------------------------------------#
    #     #   解冻后训练
    #     #------------------------------------#
    #     if Freeze_Train:
    #         if backbone == "vgg":
    #             for param in model.vgg[:28].parameters():
    #                 param.requires_grad = True
    #         elif backbone == 'mobilenetv2':
    #             for param in model.mobilenet.parameters():
    #                 param.requires_grad = True
    #
    #     for epoch in range(start_epoch, end_epoch):
    #         fit_one_epoch(model_train, model, criterion, loss_history, optimizer, epoch,
    #                 epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
    #         lr_scheduler.step()
