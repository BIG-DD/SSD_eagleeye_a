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
from onnx import numpy_helper


def get_instance(module, name, config, *args):
	return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def pth2onnx(config, pth_path, onnx_path, height, width):
    do_simplify = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = json.load(open(config))
    model = get_instance(module_arch, 'arch', config)
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    model.eval()

    # Input
    inputs = torch.randn(1, 3, height, width)
    torch.onnx.export(model,
                      inputs,
                      onnx_path,
                      verbose=False,
                      opset_version=9,
                      input_names=['data'],
                      output_names=['seg_out'])
    print('convert', onnx_path, 'to onnx finish!!!')

    model_onnx = onnx.load(onnx_path)  # load onnx model
    weights, names = [], []
    for t in model_onnx.graph.initializer:
        weights.append(numpy_helper.to_array(t))
        names.append(t.name)

    onnx_weight = dict()
    for name, weight in zip(names, weights):
        onnx_weight[name] = f'max: {weight.max()} min: {weight.min()} shape: {weight.shape}'
    for key, val in onnx_weight.items():
        print(key)
        print(val)
    onnx.checker.check_model(model_onnx)  # check onnx model
    print(onnx.helper.printable_graph(model_onnx.graph))  # print

    if do_simplify:
        print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
        model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_path)

    try:
        sess = ort.InferenceSession(onnx_path)

        for ii in sess.get_inputs():
            print("Input: ", ii)
        for oo in sess.get_outputs():
            print("Output: ", oo)

        print('read onnx using onnxruntime sucess')
    except Exception as e:
        print('read failed')
        raise e


class segmentation():
    def __init__(self, onnx_path, height, width, num_classes):
        self.inpWidth = height
        self.inpHeight = width
        self.net = cv2.dnn.readNet(onnx_path)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.01712475, 0.01750700, 0.01742919], dtype=np.float32).reshape(1, 1, 3)
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
        # img = img.astype(np.float32) / 255.0
        img = (img - self.mean) * self.std
        return img

    def segment(self, srcimg):
        # img, newh, neww, padh, padw = self.resize_image(srcimg)
        img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        # img = self._normalize(img)
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


def onnx_test(onnx_path, imgpath, height, width):

    yolonet = segmentation(onnx_path, height, width, 2)
    srcimg = cv2.imread(imgpath)
    outimg = yolonet.segment(srcimg)
    cv2.imshow('origin', srcimg)
    cv2.imshow('segment', outimg)
    cv2.waitKey()
    cv2.imwrite(imgpath.replace('.jpg', 'result.jpg'), outimg)


if __name__ == "__main__":
    height = 256
    width = 256
    config = '/media/byd/A264AC9264AC6AAD/DataSet/12_7_data/limit_conv_weight/parking_slot/1215_094939/config.json'
    pth_path = '/media/byd/A264AC9264AC6AAD/DataSet/12_7_data/eps_1e1/parking_slot/1228_094645/300.pth'
    onnx_path = pth_path.replace('.pth', f'_{height}_{width}.onnx')

    pth2onnx(config, pth_path, onnx_path, height, width)

    # imgpath = "/media/byd/A264AC9264AC6AAD/DataSet/8_25_dataB/img/04-19_ps5_2022.4.2_CZ-T-N1_97.jpg"
    # onnx_test(onnx_path, imgpath, height, width)




