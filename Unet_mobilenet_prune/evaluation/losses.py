# ------------------------------------------------------------------------------
#   Libraries
# ------------------------------------------------------------------------------
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
#   Fundamental losses
# ------------------------------------------------------------------------------

def dice_ce_loss(logits, targets, smooth=1.0):
    """
		logits: (torch.float32)  shape (N, C, H, W)
		targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
		"""
    outputs_dice = F.softmax(logits, dim=1)
    targets_dice = torch.unsqueeze(targets, dim=1)
    # targets = torch.zeros_like(logits).scatter_(dim=1, index=targets.type(torch.int64), src=torch.tensor(1.0))
    targets_dice = torch.zeros_like(logits).scatter_(dim=1, index=targets_dice.type(torch.int64),
                                                     src=torch.ones_like(logits))

    inter = outputs_dice * targets_dice
    dice = 1 - ((2 * inter.sum(dim=(2, 3)) + smooth) / (
                outputs_dice.sum(dim=(2, 3)) + targets_dice.sum(dim=(2, 3)) + smooth))
    dice_loss = dice.mean()

    targets_ce = targets.type(torch.int64)
    ce_loss = F.cross_entropy(logits, targets_ce)
    dice_ce_and_loss = 0.5 * dice_loss + 0.5 * ce_loss
    return dice_ce_and_loss


def dice_and_ce_loss(logits, targets, smooth=1.0):
    targets_ce = targets.type(torch.int64)
    ce_loss = F.cross_entropy(logits, targets_ce)
    outputs_dice = F.softmax(logits, dim=1)
    targets_dice = torch.unsqueeze(targets, dim=1)
    # targets = torch.zeros_like(logits).scatter_(dim=1, index=targets.type(torch.int64), src=torch.tensor(1.0))
    targets_dice = torch.zeros_like(logits).scatter_(dim=1, index=targets_dice.type(torch.int64),
                                                     src=torch.ones_like(logits))

    inter_dice = outputs_dice * targets_dice
    dice = 1 - ((2 * inter_dice.sum(dim=(2, 3)) + smooth) / (
                outputs_dice.sum(dim=(2, 3)) + targets_dice.sum(dim=(2, 3)) + smooth))
    dice[:, 1] = dice[:, 1]*1.5
    return dice.mean() * 0.5 + ce_loss * 0.5


def dice_loss(logits, targets, smooth=1.0):
    """
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
    outputs = F.softmax(logits, dim=1)  # 对预测值做softmax计算
    targets = torch.unsqueeze(targets, dim=1)  # 标签是3通道，增加一个通道，方便后续计算。
    targets = torch.zeros_like(logits).scatter_(dim=1, index=targets.type(torch.int64), src=torch.ones_like(
        logits))  # target标签中用1，2，3...分别代表第几类分割标签，现通过通道数表示标签类别

    inter = outputs * targets  # 计算两个标签的交集
    dice = 1 - ((2 * inter.sum(dim=(2, 3)) + smooth) / (outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth))
    dice[:, 1] = dice[:, 1]*1.5
    # return dice_sum/(targets_sum + 1e-6)
    return dice.mean()


def dice_loss_with_sigmoid(sigmoid, targets, smooth=1e-5):
    """
	sigmoid: (torch.float32)  shape (N, 1, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1}
	"""
    sigmoid = F.sigmoid(sigmoid)
    outputs = torch.squeeze(sigmoid, dim=1)
    inter = outputs * targets
    dice = 1 - ((2 * inter.sum(dim=(1, 2)) + smooth) / (outputs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + smooth))
    dice = dice.mean()
    return dice


def ce_loss(logits, targets):
    """
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
    targets = targets.type(torch.int64)
    ce_loss = F.cross_entropy(logits, targets, ignore_index=0)
    return ce_loss


# ------------------------------------------------------------------------------
#   Custom loss for BiSeNet
# ------------------------------------------------------------------------------
def custom_bisenet_loss(logits, targets):
    """
	logits: (torch.float32) (main_out, feat_os16_sup, feat_os32_sup) of shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
    if type(logits) == tuple:
        main_loss = ce_loss(logits[0], targets)
        os16_loss = ce_loss(logits[1], targets)
        os32_loss = ce_loss(logits[2], targets)
        return main_loss + os16_loss + os32_loss
    else:
        return ce_loss(logits, targets)


# ------------------------------------------------------------------------------
#   Custom loss for PSPNet
# ------------------------------------------------------------------------------
def custom_pspnet_loss(logits, targets, alpha=0.4):
    """
	logits: (torch.float32) (main_out, aux_out) of shape (N, C, H, W), (N, C, H/8, W/8)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
    if type(logits) == tuple:
        with torch.no_grad():
            _targets = torch.unsqueeze(targets, dim=1)
            aux_targets = F.interpolate(_targets, size=logits[1].shape[-2:], mode='bilinear', align_corners=True)[:, 0,
                          ...]

        main_loss = ce_loss(logits[0], targets)
        aux_loss = ce_loss(logits[1], aux_targets)
        return main_loss + alpha * aux_loss
    else:
        return ce_loss(logits, targets)


# ------------------------------------------------------------------------------
#   Custom loss for ICNet
# ------------------------------------------------------------------------------
def custom_icnet_loss(logits, targets, alpha=[0.4, 0.16]):
    """
	logits: (torch.float32)
		[train_mode] (x_124_cls, x_12_cls, x_24_cls) of shape
						(N, C, H/4, W/4), (N, C, H/8, W/8), (N, C, H/16, W/16)

		[valid_mode] x_124_cls of shape (N, C, H, W)

	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
    if type(logits) == tuple:
        with torch.no_grad():
            targets = torch.unsqueeze(targets, dim=1)
            target1 = F.interpolate(targets, size=logits[0].shape[-2:], mode='bilinear', align_corners=True)[:, 0, ...]
            target2 = F.interpolate(targets, size=logits[1].shape[-2:], mode='bilinear', align_corners=True)[:, 0, ...]
            target3 = F.interpolate(targets, size=logits[2].shape[-2:], mode='bilinear', align_corners=True)[:, 0, ...]

        loss1 = ce_loss(logits[0], target1)
        loss2 = ce_loss(logits[1], target2)
        loss3 = ce_loss(logits[2], target3)
        return loss1 + alpha[0] * loss2 + alpha[1] * loss3

    else:
        return ce_loss(logits, targets)


def OHEM_cross_entropy(logits, targets, epoch, smooth=1e-5):

    # dice_loss
    outputs_dice = F.softmax(logits, dim=1)
    targets_dice = torch.unsqueeze(targets, dim=1)
    # targets = torch.zeros_like(logits).scatter_(dim=1, index=targets.type(torch.int64), src=torch.tensor(1.0))
    targets_dice = torch.zeros_like(logits).scatter_(dim=1, index=targets_dice.type(torch.int64),
                                                     src=torch.ones_like(logits))

    inter = outputs_dice * targets_dice
    dice = 1 - ((2 * inter.sum(dim=(2, 3)) + smooth) / (
                outputs_dice.sum(dim=(2, 3)) + targets_dice.sum(dim=(2, 3)) + smooth))
    dice_one = dice[:, 0].mean()
    dice_two = dice[:, 1].mean()
    back_num = targets_dice[:, 0, :, :].sum()
    fore_num = targets_dice[:, 1, :, :].sum()
    # dice_new_loss = ((dice_two/(dice_one+smooth))*dice_two*0.5 + dice_one)*0.5
    # dice_loss = dice.mean()
    # print('dice_two/(dice_one+smooth)', dice_two/(dice_one+smooth))
    dice_loss = (dice_one + dice_two)*0.5

    # new_ohem_loss
    input_prob = F.softmax(logits, dim=1) # (64, 2, 256, 256)
    logits_new = abs(input_prob[:, 0, :, :] - input_prob[:, 1, :, :])
    th = 0.8

    if epoch <= 25:
        th = 0.8
    elif epoch <= 45:
        th = 0.75
    elif epoch <= 65:
        th = 0.7
    elif epoch <= 85:
        th = 0.65
    elif epoch <= 105:
        th = 0.6

    th = 1
    logits_new_mask = logits_new > th
    targets_new = copy.deepcopy(targets)
    targets_new[logits_new_mask]=255
    foreground_mask = targets_new==1
    background_mask = targets_new==0
    # new_targets = generate_new_target(input_prob, targets)
    loss = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    targets_new = targets_new.long().cuda(targets.get_device())
    OHEM_loss = loss(logits, targets_new) # (64, 256, 256)


    OHEM_fore_ground_loss = OHEM_loss*targets
    # print('targets[foreground_mask].sum()',targets[foreground_mask].sum())
    OHEM_fore_ground_loss_mean = OHEM_fore_ground_loss.sum()/targets[foreground_mask].sum()
    OHEM_back_ground_loss = OHEM_loss*(1-targets)
    OHEM_back_ground_loss_mean = OHEM_back_ground_loss.sum()/(1-targets)[background_mask].sum()
    # print('(1-targets)[background_mask].sum()',(1-targets)[background_mask].sum())
    # print('foreground_0.4:', (targets[foreground_mask].sum()/64).data, ' background_0.4:', ((1-targets)[background_mask].sum()/64).data)

    return ((0.6*OHEM_fore_ground_loss_mean+0.4*OHEM_back_ground_loss_mean)+dice_loss)*0.5, OHEM_fore_ground_loss_mean, OHEM_back_ground_loss_mean


def generate_new_target(predict, target):
    ignore_label = 255
    np_predict = predict.data.cpu().numpy()  # shape(64, 2, 256, 256)

    # img
    # predict_one = predict[0, 1, :, :].data
    # predict_one = np.array(predict_one.cpu())
    # predict_one = predict_one.flatten()
    # xs = np.linspace(0, 256*256,256*256)
    # plt.figure(figsize=(14,8))
    # plt.plot(xs, predict_one)
    # plt.show()

    np_target = target.data.cpu().numpy()  # shape(64, 256, 256)
    n, c, h, w = np_predict.shape  # (64, 2, 256, 256)

    threshold = find_threshold(np_predict, np_target)  # 寻找阈值0.7
    if threshold != 0.7:
        print('threshold', threshold)

    input_label = np_target.ravel().astype(np.int32)  # shape(4194304)
    input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))  # (2, 4194304)

    valid_flag = input_label != ignore_label  # label中有效位置(4194304)
    valid_inds = np.where(valid_flag)[0]  # (4194304)
    label = input_label[valid_flag]  # 一次筛选：不为255的label(4194304)
    num_valid = valid_flag.sum()  # 4194304

    if num_valid > 0:
        prob = input_prob[:, valid_flag]  # (2, 4194304)
        pred = prob[label, np.arange(len(label), dtype=np.int32)]  # 不明白这一步的操作??? (4194304)
        kept_flag = pred <= threshold  # 二次筛选：在255中找出pred≤0.7的位置
        valid_inds = valid_inds[kept_flag]  # shape(3424872)

    label = input_label[valid_inds].copy()  # 从原label上扣下来shape(3424872)
    input_label.fill(ignore_label)  # shape(4194304)每个值都为255
    input_label[valid_inds] = label  # 把二次筛选后有效区域的对应位置为label，其余为255
    new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(
        target.get_device())  # torch.Size([64, 256, 256])

    return new_target  # torch.Size([64, 256, 256])


def find_threshold(np_predict, np_target):
    # downsample 1/8

    factor = 8  # 8
    min_kept = 100000
    ignore_label = 255
    thresh = 0.7

    predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)  # 双线性插值  shape(64, 2, 32, 32)
    target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)  # 最近临插值  shape(64, 32, 32)

    n, c, h, w = predict.shape  # (64, 2, 32, 32)
    min_kept = min_kept // (factor * factor)  # int(self.min_kept_ratio * n * h * w)   #100000/64 = 1562

    input_label = target.ravel().astype(np.int32)  # 将多维数组转化为一维 shape(65536)
    input_prob = np.rollaxis(predict, 1).reshape((c, -1))  # 轴1滚动到轴0、shape((2, 65536))

    valid_flag = input_label != ignore_label  # label中有效位置(9216, )
    valid_inds = np.where(valid_flag)[0]  # (9013, )
    label = input_label[valid_flag]  # 有效label(9013, )
    num_valid = valid_flag.sum()  # 9013
    if min_kept >= num_valid:  # 1562 >= 9013
        threshold = 1.0
    elif num_valid > 0:  # 9013 > 0
        prob = input_prob[:, valid_flag]  # (2, 65536) #找出有效区域对应的prob
        pred = prob[label, np.arange(len(label), dtype=np.int32)]  # ???    shape(65536)
        threshold = thresh  # 0.7
        if min_kept > 0:  # 1562>0
            # k_th = min(len(pred), min_kept) - 1  # min(9013, 1562)-1 = 1561
            k_th = int(len(pred) * (1 / 2))
            new_array = np.partition(pred, k_th)  # 排序并分成两个区，小于第1561个及大于第1561个
            new_threshold = new_array[k_th]  # 第1561对应的pred 0.03323581
            if new_threshold > thresh:  # 返回的阈值只能≥0.7
                threshold = new_threshold
    return threshold


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.5,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_focal'):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal
                Loss. Defaults to 0.5. When a list is provided, the length
                of the list should be equal to the number of classes.
                Please be careful that this parameter is not the
                class-wise weight but the weight of a binary classification
                problem. This binary classification problem regards the
                pixels which belong to one class as the foreground
                and the other pixels as the background, each element in
                the list is the weight of the corresponding foreground class.
                The value of alpha or each element of alpha should be a float
                in the interval [0, 1]. If you want to specify the class-wise
                weight, please use `class_weight` parameter.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_focal'.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, \
            'AssertionError: Only sigmoid focal loss supported now.'
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        assert isinstance(class_weight, list) or class_weight is None, \
            'AssertionError: class_weight must be None or of type list'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape
                (N, C) where C = number of classes, or
                (N, C, d_1, d_2, ..., d_K) with K≥1 in the
                case of K-dimensional loss.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1,
                or (N, d_1, d_2, ..., d_K) with K≥1 in the case of
                K-dimensional loss. If containing class probabilities,
                same shape as the input.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used
                to override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            ignore_index (int, optional): The label index to be ignored.
                Default: 255
        Returns:
            torch.Tensor: The calculated loss
        """
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
               "The shape of pred doesn't match the shape of target"

        original_shape = pred.shape

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            # target with shape [B, C, d_1, d_2, ...]
            # transform it's shape into [N, C]
            # [B, C, d_1, d_2, ...] -> [C, B, d_1, d_2, ..., d_k]
            target = target.transpose(0, 1)
            # [C, B, d_1, d_2, ..., d_k] -> [C, N]
            target = target.reshape(target.size(0), -1)
            # [C, N] -> [N, C]
            target = target.transpose(0, 1).contiguous()
        else:
            # target with shape [B, d_1, d_2, ...]
            # transform it's shape into [N, ]
            target = target.view(-1).contiguous()
            valid_mask = (target != ignore_index).view(-1, 1)
            # avoid raising error when using F.one_hot()
            target = torch.where(target == ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            if torch.cuda.is_available() and pred.is_cuda:
                if target.dim() == 1:
                    one_hot_target = F.one_hot(target, num_classes=num_classes)
                else:
                    one_hot_target = target
                    target = target.argmax(dim=1)
                    valid_mask = (target != ignore_index).view(-1, 1)
                calculate_loss_func = sigmoid_focal_loss
            else:
                one_hot_target = None
                if target.dim() == 1:
                    target = F.one_hot(target, num_classes=num_classes)
                else:
                    valid_mask = (target.argmax(dim=1) != ignore_index).view(
                        -1, 1)
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                one_hot_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                class_weight=self.class_weight,
                valid_mask=valid_mask,
                reduction=reduction,
                avg_factor=avg_factor)

            if reduction == 'none':
                # [N, C] -> [C, N]
                loss_cls = loss_cls.transpose(0, 1)
                # [C, N] -> [C, B, d1, d2, ...]
                # original_shape: [B, C, d1, d2, ...]
                loss_cls = loss_cls.reshape(original_shape[1],
                                            original_shape[0],
                                            *original_shape[2:])
                # [C, B, d1, d2, ...] -> [B, C, d1, d2, ...]
                loss_cls = loss_cls.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name