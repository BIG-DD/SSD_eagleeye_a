import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import long
import math
import numpy as np


class MultiboxLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = torch.FloatTensor([negatives_for_hard])[0]

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = torch.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = torch.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
        return torch.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-7)
        softmax_loss = -torch.sum(y_true * torch.log(y_pred), axis=-1)
        return softmax_loss

    def forward(self, y_true, y_pred):
        # --------------------------------------------- #
        #   y_true batch_size, 8732, 4 + self.num_classes + 1
        #   y_pred batch_size, 8732, 4 + self.num_classes
        # --------------------------------------------- #
        num_boxes       = y_true.size()[1]
        y_pred          = torch.cat([y_pred[0], nn.Softmax(-1)(y_pred[1])], dim = -1)

        # --------------------------------------------- #
        #   分类的loss
        #   batch_size,8732,21 -> batch_size,8732
        # --------------------------------------------- #
        conf_loss = self._softmax_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:])
        
        # --------------------------------------------- #
        #   框的位置的loss
        #   batch_size,8732,4 -> batch_size,8732
        # --------------------------------------------- #
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # --------------------------------------------- #
        #   获取所有的正标签的loss
        # --------------------------------------------- #
        pos_loc_loss = torch.sum(loc_loss * y_true[:, :, -1], axis=1)
        pos_conf_loss = torch.sum(conf_loss * y_true[:, :, -1], axis=1)

        # --------------------------------------------- #
        #   每一张图的正样本的个数
        #   num_pos     [batch_size,]
        # --------------------------------------------- #
        num_pos = torch.sum(y_true[:, :, -1], axis=-1)

        # --------------------------------------------- #
        #   每一张图的负样本的个数
        #   num_neg     [batch_size,]
        # --------------------------------------------- #
        num_neg = torch.min(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        # 找到了哪些值是大于0的
        pos_num_neg_mask = num_neg > 0
        # --------------------------------------------- #
        #   如果所有的图，正样本的数量均为0
        #   那么则默认选取100个先验框作为负样本
        # --------------------------------------------- #
        has_min = torch.sum(pos_num_neg_mask)
        
        # --------------------------------------------- #
        #   从这里往后，与视频中看到的代码有些许不同。
        #   由于以前的负样本选取方式存在一些问题，
        #   我对该部分代码进行重构。
        #   求整个batch应该的负样本数量总和
        # --------------------------------------------- #
        num_neg_batch = torch.sum(num_neg) if has_min > 0 else self.negatives_for_hard

        # --------------------------------------------- #
        #   对预测结果进行判断，如果该先验框没有包含物体
        #   那么它的不属于背景的预测概率过大的话
        #   就是难分类样本
        # --------------------------------------------- #
        confs_start = 4 + self.background_label_id + 1
        confs_end   = confs_start + self.num_classes - 1

        # --------------------------------------------- #
        #   batch_size,8732
        #   把不是背景的概率求和，求和后的概率越大
        #   代表越难分类。
        # --------------------------------------------- #
        max_confs = torch.sum(y_pred[:, :, confs_start:confs_end], dim=2)

        # --------------------------------------------------- #
        #   只有没有包含物体的先验框才得到保留
        #   我们在整个batch里面选取最难分类的num_neg_batch个
        #   先验框作为负样本。
        # --------------------------------------------------- #
        max_confs   = (max_confs * (1 - y_true[:, :, -1])).view([-1])

        _, indices  = torch.topk(max_confs, k = int(num_neg_batch.cpu().numpy().tolist()))

        neg_conf_loss = torch.gather(conf_loss.view([-1]), 0, indices)

        # 进行归一化
        num_pos     = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))
        total_loss  = torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        total_loss  = total_loss / torch.sum(num_pos)
        total_pos_loc_loss = torch.sum(self.alpha * pos_loc_loss) / torch.sum(num_pos)
        total_pos_conf_loss = torch.sum(pos_conf_loss) / torch.sum(num_pos)
        total_neg_conf_loss = torch.sum(neg_conf_loss) / torch.sum(num_pos)

        return total_loss, total_pos_loc_loss, total_pos_conf_loss, total_neg_conf_loss


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.  [bs,24564] 所有anchor属于背景的损失
        labels (N, num_priors): the labels. 所有anchor的真实框类别标签 0表示背景类 [bs, 24564]
        neg_pos_ratio:  the ratio between the negative examples and positive examples. 负样本:正样本=3:1
    """
    pos_mask = labels > 0  # 正类True 负类背景类False
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)   # [1个,25个] 第一张图1个正样本 第二张图25个正样本
    num_neg = num_pos * neg_pos_ratio  # [3, 75] 第一张图3个负样本 第二张图75个负样本

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)  # 对负样本损失进行排序
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg  # 选出损失最高的前num_neg个负样本
    return pos_mask | neg_mask


# 稀疏化
def updateBN(model, sparsity=1e-4):
    for k,m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(sparsity*torch.sign(m.weight.data))

