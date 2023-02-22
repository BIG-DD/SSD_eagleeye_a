import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from utils.utils import augments

class SSDDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, augment, overlap_threshold=0.5):
        super(SSDDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.resize_shape        = input_shape
        self.anchors            = anchors
        self.num_anchors        = len(anchors)
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.augment              = augment
        self.overlap_threshold  = overlap_threshold

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, boxes = self.get_random_data(self.annotation_lines[index], self.resize_shape, augment=self.augment)
        boxes_one_hot = boxes
        # image_data  = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        image_data = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        if len(boxes)!=0:
            box               = np.array(boxes[:, :4], dtype=np.float32)
            # 进行归一化，调整到0-1之间
            box[:, [0, 2]]    = box[:, [0, 2]] / self.resize_shape[1]
            box[:, [1, 3]]    = box[:, [1, 3]] / self.resize_shape[0]
            # 对真实框的种类进行one hot处理
            one_hot_label   = np.eye(self.num_classes - 1)[np.array(boxes[:, 4], np.int32)]
            boxes_one_hot   = np.concatenate([box, one_hot_label], axis=-1)
        # 已对label进行了编码
        boxes_one_hot = self.assign_boxes(boxes_one_hot)

        return np.array(image_data), np.array(boxes_one_hot), np.array(boxes)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, resize_shape, augment=True):
        line = annotation_line.split()
        #   读取图像并转换成RGB图像
        img = cv2.imread(line[0])#Image.open(line[0])
        #   获得标注框
        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        ih, iw, _ = img.shape
        if ih != 512 or iw != 512:
            rate_x = 512.0 / iw
            rate_y = 512.0 / ih
            img = cv2.resize(img, (512, 512))
            for j, box in enumerate(boxes):
                boxes[j][0] = int(box[0] * rate_x)
                boxes[j][1] = int(box[1] * rate_y)
                boxes[j][2] = int(box[2] * rate_x)
                boxes[j][3] = int(box[3] * rate_y)
                # cv2.rectangle(img, (boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3]), (0, 0, 255), 3)
            # cv2.imshow('img', img)
            # cv2.waitKey()
        #   对图像进行augment
        if augment:
            img, boxes = augments(img, boxes)

        #   获得图像的高宽与目标高宽
        ih, iw, _ = img.shape
        resize_H, resize_w = resize_shape
        scale_w, scale_h = 1.0 * resize_w / iw, 1.0 * resize_H / ih
        img = cv2.resize(img, (resize_w, resize_H))
        #   对真实框进行调整
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            # print(boxes)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
            boxes[:, 2][boxes[:, 2] > resize_w] = resize_w
            boxes[:, 3][boxes[:, 3] > resize_H] = resize_H
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]

        return img, boxes

    def iou(self, box):
        #---------------------------------------------#
        #   计算出每个真实框与所有的先验框的iou
        #   判断真实框与先验框的重合情况
        #---------------------------------------------#
        inter_upleft    = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright  = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh    = inter_botright - inter_upleft
        inter_wh    = np.maximum(inter_wh, 0)
        inter       = inter_wh[:, 0] * inter_wh[:, 1]
        #---------------------------------------------# 
        #   真实框的面积
        #---------------------------------------------#
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        #---------------------------------------------#
        #   先验框的面积
        #---------------------------------------------#
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0])*(self.anchors[:, 3] - self.anchors[:, 1])
        #---------------------------------------------#
        #   计算iou
        #---------------------------------------------#
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):
        #---------------------------------------------#
        #   计算当前真实框和先验框的重合情况
        #   iou [self.num_anchors]
        #   encoded_box [self.num_anchors, 5]
        #---------------------------------------------#
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        
        #---------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #   真实框可以由这个先验框来负责预测
        #---------------------------------------------#
        assign_mask = iou > self.overlap_threshold

        #---------------------------------------------#
        #   如果没有一个先验框重合度大于self.overlap_threshold
        #   则选择重合度最大的为正样本
        #---------------------------------------------#
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        
        #---------------------------------------------#
        #   利用iou进行赋值 , 大于阈值或者iou最大的才赋值，其他的全为0
        #---------------------------------------------#
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        #---------------------------------------------#
        #   找到对应的先验框
        #---------------------------------------------#
        assigned_anchors = self.anchors[assign_mask]

        #---------------------------------------------#
        #   逆向编码，将真实框转化为ssd预测结果的格式
        #   先计算真实框的中心与长宽
        #---------------------------------------------#
        box_center  = 0.5 * (box[:2] + box[2:])
        box_wh      = box[2:] - box[:2]
        #---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        #---------------------------------------------#
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh     = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])
        
        #------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        #------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        #---------------------------------------------------#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4:-1    的内容为先验框所对应的种类，默认为背景
        #   -1      的内容为当前先验框是否包含目标
        #---------------------------------------------------#
        # self.num_classes 是背景+类别  +1 是表示anchor中是否有目标
        assignment          = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        # 默认所有anchor为背景
        assignment[:, 4]    = 1.0
        if len(boxes) == 0:
            return assignment

        # 对每一个真实框都进行iou计算
        encoded_boxes   = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        #---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_anchors, 4 + 1]
        #   4是编码后的结果，1为iou
        #---------------------------------------------------#
        encoded_boxes   = encoded_boxes.reshape(-1, self.num_anchors, 5)
        
        #---------------------------------------------------#
        #   [num_anchors]求取每一个先验框重合度最大的真实框
        #---------------------------------------------------#
        best_iou        = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx    = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask   = best_iou > 0
        best_iou_idx    = best_iou_idx[best_iou_mask]
        
        #---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        #---------------------------------------------------#
        assign_num      = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes   = encoded_boxes[:, best_iou_mask, :]
        #---------------------------------------------------#
        #   编码后的真实框的赋值
        #---------------------------------------------------#
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        #----------------------------------------------------------#
        #   4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        #----------------------------------------------------------#
        assignment[:, 4][best_iou_mask]     = 0
        assignment[:, 5:-1][best_iou_mask]  = boxes[best_iou_idx, 4:]
        #----------------------------------------------------------#
        #   -1表示先验框是否有对应的物体
        #----------------------------------------------------------#
        assignment[:, -1][best_iou_mask]    = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment


# DataLoader中collate_fn使用
def ssd_dataset_collate(batch):
    images, bboxes, boxes = [], [], []
    for img, box_one_hot, box in batch:
        images.append(img)
        bboxes.append(box_one_hot)
        boxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    boxes = np.array(boxes)

    return images, bboxes, boxes
