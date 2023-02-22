import numpy as np
import torch
from torch import nn
from torchvision.ops import nms


class BBoxUtility(object):
    def __init__(self, class_names, num_classes, input_shape, image_shape, anchors, cuda):
        self.class_names = class_names
        self.num_classes    = num_classes
        self.input_shape    = input_shape
        self.image_shape    = image_shape
        self.anchors = anchors
        self.nms_iou        = 0.3
        self.confidence     = 0.1
        self.letterbox_image = False
        if cuda:
            self.anchors = torch.from_numpy(self.anchors).type(torch.FloatTensor).cuda()

    def ssd_correct_boxes(self, box_xy, box_wh):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(self.input_shape)
        image_shape = np.array(self.image_shape)

        if self.letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        # boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes  = np.concatenate([box_mins[..., 1:2], box_mins[..., 0:1], box_maxes[..., 1:2], box_maxes[..., 0:1]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def decode_boxes(self, mbox_loc, variances):
        # 获得先验框的宽与高
        anchor_width     = self.anchors[:, 2] - self.anchors[:, 0]
        anchor_height    = self.anchors[:, 3] - self.anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x  = 0.5 * (self.anchors[:, 2] + self.anchors[:, 0])
        anchor_center_y  = 0.5 * (self.anchors[:, 3] + self.anchors[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[0]
        decode_bbox_center_y += anchor_center_y
        
        # 真实框的宽与高的求取
        decode_bbox_width   = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width   *= anchor_width
        decode_bbox_height  = torch.exp(mbox_loc[:, 3] * variances[1])
        decode_bbox_height  *= anchor_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), dim=-1)
        # 防止超出0与1
        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        return decode_bbox

    def decode_box(self, predictions, variances=[0.1, 0.2]):
        #---------------------------------------------------#
        #   :4是回归预测结果
        #---------------------------------------------------#
        mbox_loc        = predictions[0]    # torch.tensor(predictions[0])
        #---------------------------------------------------#
        #   获得种类的置信度
        #---------------------------------------------------#
        mbox_conf       = nn.Softmax(-1)(predictions[1])

        results = []
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(len(mbox_loc)):
            results.append([])
            #--------------------------------#
            #   利用回归结果对先验框进行解码
            #--------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], variances)

            for c in range(1, self.num_classes):
                #--------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                #--------------------------------#
                c_confs     = mbox_conf[i, :, c]
                c_confs_m   = c_confs > self.confidence
                if len(c_confs[c_confs_m]) > 0:
                    #-----------------------------------------#
                    #   取出得分高于confidence的框
                    #-----------------------------------------#
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        self.nms_iou
                    )
                    #-----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    #-----------------------------------------#
                    good_boxes  = boxes_to_process[keep]
                    confs       = confs_to_process[keep][:, None]
                    labels      = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    #-----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    #-----------------------------------------#
                    c_pred      = torch.cat((good_boxes, labels, confs), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh)

        return results

    def box_iou(self, box0, box1):
        inter_upleft = np.maximum(box0[:2], box1[:2])
        inter_botright = np.minimum(box0[2:], box1[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[0] * inter_wh[1]
        # ---------------------------------------------#
        #   真实框的面积
        # ---------------------------------------------#
        area_true = (box1[2] - box1[0]) * (box1[3] - box1[1])
        # ---------------------------------------------#
        #   先验框的面积
        # ---------------------------------------------#
        area_gt = (box0[2] - box0[0]) * (box0[3] - box0[1])
        # ---------------------------------------------#
        #   计算iou
        # ---------------------------------------------#
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def get_ap(self, preds, targets, TP, PREDS, P_N, nms_iou=0.5):
        # TP, PREDS, P_N = {}, {}, {}
        for b in range(len(preds)):
            pred = preds[b]
            target = targets[b]
            for i in range(len(pred)):
                pr = pred[i]
                for j in range(len(target)):
                    gt = target[j]
                    if pr[4] == gt[4]:
                        box1, box2 = [], []
                        for k in range(4):
                            box1.append(pr[k])
                            box2.append(gt[k])
                        box1, box2 = np.array(box1), np.array(box2)
                        iou = self.box_iou(box1, box2)
                        if iou > nms_iou:
                            if TP.get(int(pr[4])):  # 查找key
                                TP[int(pr[4])] += 1
                            else:
                                TP.update({int(pr[4]): 1})  # 增加元素

            for j in range(len(pred)):
                pr = int(pred[j][4])
                if PREDS.get(pr):  # 查找key
                    PREDS[pr] += 1
                else:
                    PREDS.update({pr: 1})  # 增加元素
            #
            for j in range(len(target)):
                gt = int(target[j][4])
                if P_N.get(gt):  # 查找key
                    P_N[gt] += 1
                else:
                    P_N.update({gt: 1})  # 增加元素

        return TP, PREDS, P_N

    def print_recall_precision(self, TP, PREDS, P_N):
        Recall = np.zeros(self.num_classes - 1)
        Precision = np.zeros(self.num_classes - 1)
        for key, P_val in P_N.items():
            TP_val = 0
            if TP.get(key):
                TP_val = TP[key]
            Recall[key] += 1.0 * TP_val / P_val

        for key, PR_val in PREDS.items():
            TP_val = 0
            if TP.get(key):
                TP_val = TP[key]
            Precision[key] += 1.0 * TP_val / PR_val
        Recall_str, Precision_str = 'Recall   ', 'Precision'
        for i, class_name in enumerate(self.class_names):
            Recall_str += ' ' + class_name + ':' + str(Recall[i])[:6]
            Precision_str += ' ' + class_name + ':' + str(Precision[i])[:6]
        print(Recall_str)
        print(Precision_str)