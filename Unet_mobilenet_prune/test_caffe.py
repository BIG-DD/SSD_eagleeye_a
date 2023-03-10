import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import models as module_arch
import json



class segmentation():
    def __init__(self, param_path, weight_path, height, width, num_classes):
        self.inpWidth = height
        self.inpHeight = width
        self.net = cv2.dnn.readNet(param_path, weight_path)
        # self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        self.mean = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(1, 1, 3)
        self.keep_ratio = True

    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, padh, padw

    def _normalize(self, img):  ### c++: https://blog.csdn.net/wuqingshan2010/article/details/107727909
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) * self.std
        # img = (img - self.mean)
        return img

    def segment(self, srcimg):
        # img, newh, neww, padh, padw = self.resize_image(srcimg)
        # img = self._normalize(img)
        img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(img)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # inference output
        outimg = srcimg.copy()
        lane_line_mask = outs[0]#[:, padh:(self.inpHeight - padh), padw:(self.inpWidth - padw)]
        lane_line_mask = lane_line_mask[0]
        seg_id = np.argmax(lane_line_mask, axis=0).astype(np.uint8)
        seg_id = cv2.resize(seg_id, (srcimg.shape[1], srcimg.shape[0]), interpolation=cv2.INTER_NEAREST)
        outimg[seg_id == 1] = [255, 0, 0]

        return outimg


def predict(param_path, weight_path, imgpath, height, width):
    yolonet = segmentation(param_path, weight_path, height, width, 2)

    srcimg = cv2.imread(imgpath)
    # srcimg = srcimg[..., ::-1]
    outimg = yolonet.segment(srcimg)
    cv2.imwrite(imgpath.replace('.jpg', 'result_mobile.jpg'), outimg)
    # cv2.imshow('origin', srcimg)
    # cv2.imshow('segment', outimg)
    # cv2.waitKey()



if __name__ == "__main__":
    height = 256
    width = 256
    # root = './workspace/checkpoints/parking_slot/0518_215729_MCnet_resnet18/'
    root = './workspace/checkpoints/parking_slot/0522_121259/'
    param_path = root + '180.prototxt'
    weight_path = root + '180.caffemodel'

    # imgpath = "/media/z590/D/PublicDataSet/BYD_parking_slot/train_val_dataset/train_part1_part2_val_part3_1/images/val/04-19_ps1_2022.2.24_CZ-T-N3_10.jpg"
    imgpath = "/media/z590/D/PublicDataSet/BYD_parking_slot/pickup_incline/04-19_ps1_2022.2.24_CZ-T-N4_31.jpg"
    predict(param_path, weight_path, imgpath, height, width)




