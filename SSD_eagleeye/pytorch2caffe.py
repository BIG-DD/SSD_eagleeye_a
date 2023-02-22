import os, sys
import torch
import torch.nn as nn
from nets.ssd import SSD300, SSD_Prune
from utils.utils import get_classes
import cv2
import numpy as np
from utils.utils_bbox import BBoxUtility
from utils.anchors import get_anchors
import json
import base64
import math
import shutil
from nets.mobilenetv2 import MobileNetV2_4_2_prune
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET


PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def get_files(files_path, exts):
    '''
    find image files in data path
    :return: list of files found
    '''
    files = []
    for parent, dirnames, filenames in os.walk(files_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
    return files


def read_VOCxml(file_path):
    tree = ET.ElementTree(file=file_path)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    object_set = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    labels = []
    boxs = []
    for object in object_set:
        obj_name = object.find('name').text
        bnd_box = object.find('bndbox')
        x1 = int(float(bnd_box.find('xmin').text))  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(float(bnd_box.find('ymin').text))  # -1
        x2 = int(float(bnd_box.find('xmax').text))  # -1
        y2 = int(float(bnd_box.find('ymax').text))  # -1
        boxs.append([x1, y1, x2, y2])
        labels.append(obj_name)

    return labels, boxs


def read_json(file):
    points, labels, shape_types= [], [], []
    with open(file, 'r', encoding='utf-8') as f:
        setting = json.load(f)
    coordinate = setting['shapes']

    for j in range(len(coordinate)):
        shape_type = coordinate[j]['shape_type']
        shape_types.append(shape_type)
        # if shape_type != 'rectangle':
        #     continue
        label = coordinate[j]['label']
        point = coordinate[j]['points']
        labels.append(label)
        points.append(point)
        # if label == 'APZU3406625':
        #     print(file)

    return labels, points, shape_types


def creat_json(dir_path, name, labels, points, shape_types, h, w):
    with open(dir_path+name.replace('.json', '.jpg'), 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    shapes_context = []
    for i, point in enumerate(points):
        str_points = []
        # for coor in point:
        str_points.append([(math.ceil(point[0])), (math.ceil(point[1]))])
        shape_context = {
            "shape_type": shape_types[i],
            "label": labels[i],
            "line_color": None,
            "points": str_points,
            "fill_color": None
        }
        shapes_context.append(shape_context)

    json_str = {
        "flags": {},
        "imagePath": name,
        "shapes": shapes_context,
        "imageData": imageData,
        "version": "3.11.0",
        "lineColor": [0, 255, 0, 128],
        "imageHeight": h,
        "imageWidth": w,
        "fillColor": [255, 0, 0, 128]

    }

    with open(dir_path + '/' + name.replace('.jpg', '.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(json_str, indent=2, ensure_ascii=False))


def get_sparsity_BN_params(features, threshold=0.125):
    backbone = features.backbone
    for i, m in enumerate(backbone.modules()):
        if isinstance(m, nn.BatchNorm2d):
            mask = m.weight.data.abs().reshape(-1) > threshold
            mask_num = int(mask.sum())
            sum_mask = len(mask)
            print('sum:'+str(sum_mask)+'->'+'prune:'+str(sum_mask-mask_num))
    extras = features.extras
    for i, m in enumerate(extras.modules()):
        if isinstance(m, nn.BatchNorm2d):
            mask = m.weight.data.abs().reshape(-1) > threshold
            mask_num = int(mask.sum())
            sum_mask = len(mask)
            print('sum:'+str(sum_mask)+'->'+'prune:'+str(sum_mask-mask_num))
    return


def iou(box0, box1):
    inter_upleft    = np.maximum(box0[:2], box1[:2])
    inter_botright  = np.minimum(box0[2:], box1[2:])

    inter_wh    = inter_botright - inter_upleft
    inter_wh    = np.maximum(inter_wh, 0)
    inter       = inter_wh[0] * inter_wh[1]
    #---------------------------------------------#
    #   真实框的面积
    #---------------------------------------------#
    area_true = (box1[2] - box1[0]) * (box1[3] - box1[1])
    #---------------------------------------------#
    #   先验框的面积
    #---------------------------------------------#
    area_gt = (box0[2] - box0[0])*(box0[3] - box0[1])
    #---------------------------------------------#
    #   计算iou
    #---------------------------------------------#
    union = area_true + area_gt - inter

    iou = inter / union
    return iou


def pth2caffe():
    # imgpath = "/media/z590/D/DataSet/BYD_obstacle/test/0823_record0/record_3/6.jpg"
    imgpath = "/media/z590/D/DataSet/BYD_corner_point/test/04-19_ps1_2022.2.24_CZ-T-N3_4.jpg"
    root = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/loss_2023_01_17_17_45_49/'
    # root = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/'
    model_path = root + 'ep100-loss1.318-val_loss1.200.pth'
    param_path = root + 'mobilenetv2_4_tiny.prototxt'
    weight_path = root + 'mobilenetv2_4_tiny.caffemodel'
    backbone = "MobileNetV2_4_2"
    classes_path = 'data/voc_coner_point_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    # class_names = ['person','non','cone','pole','close','open']
    num_classes = num_classes + 1
    net = SSD300(num_classes, backbone, False)
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.eval()
    # get_sparsity_BN_params(net)

    print('{} model, anchors, and classes loaded.'.format(model_path))

    from pth2caffe import pytorch_to_caffe

    name = 'mobilenetv1'
    dummy_input = torch.ones([1, 3, 1024, 1024])    # 转换caffe时需要把输入图片尺寸设置为[1, 3, 1024, 1024]，在prototxt文件中在修改成[1, 3, 512, 512]
    pytorch_to_caffe.trans_net(net, dummy_input, name)
    pytorch_to_caffe.save_prototxt(param_path)
    pytorch_to_caffe.save_caffemodel(weight_path)


def pth2caffe2():
    # imgpath = "/media/z590/D/DataSet/BYD_obstacle/test/0823_record0/record_3/6.jpg"
    imgpath = "/media/z590/D/DataSet/BYD_corner_point/test/04-19_ps1_2022.2.24_CZ-T-N3_4.jpg"
    root = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/loss_2023_01_30_17_45_50_MobileNetV2_4_2_finetune/'
    # root = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/'
    model_path = root + 'ep098-loss1.237-val_loss1.196.pth'
    param_path = root + 'MobileNetV2_4_2_prune_finetune.prototxt'
    weight_path = root + 'MobileNetV2_4_2_prune_finetune.caffemodel'

    classes_path = 'data/voc_coner_point_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    num_classes = num_classes + 1
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_params = [[3, 24, (3, 3), (2, 2), (1, 1), 1],
                  [[[24, 24, (3, 3), (1, 1), (1, 1), 24], [24, 16, (1, 1), (1, 1), (0, 0), 1]], 1, False], [
                      [[16, 32, (1, 1), (1, 1), (0, 0), 1], [32, 32, (3, 3), (2, 2), (1, 1), 32],
                       [32, 24, (1, 1), (1, 1), (0, 0), 1]], 4, False], [
                      [[24, 40, (1, 1), (1, 1), (0, 0), 1], [40, 40, (3, 3), (1, 1), (1, 1), 40],
                       [40, 24, (1, 1), (1, 1), (0, 0), 1]], 4, True], [
                      [[24, 72, (1, 1), (1, 1), (0, 0), 1], [72, 72, (3, 3), (2, 2), (1, 1), 72],
                       [72, 32, (1, 1), (1, 1), (0, 0), 1]], 4, False], [
                      [[32, 72, (1, 1), (1, 1), (0, 0), 1], [72, 72, (3, 3), (1, 1), (1, 1), 72],
                       [72, 32, (1, 1), (1, 1), (0, 0), 1]], 4, True], [
                      [[32, 72, (1, 1), (1, 1), (0, 0), 1], [72, 72, (3, 3), (1, 1), (1, 1), 72],
                       [72, 32, (1, 1), (1, 1), (0, 0), 1]], 4, True], [
                      [[32, 88, (1, 1), (1, 1), (0, 0), 1], [88, 88, (3, 3), (2, 2), (1, 1), 88],
                       [88, 64, (1, 1), (1, 1), (0, 0), 1]], 4, False], [
                      [[64, 200, (1, 1), (1, 1), (0, 0), 1], [200, 200, (3, 3), (1, 1), (1, 1), 200],
                       [200, 64, (1, 1), (1, 1), (0, 0), 1]], 4, True], [
                      [[64, 152, (1, 1), (1, 1), (0, 0), 1], [152, 152, (3, 3), (1, 1), (1, 1), 152],
                       [152, 64, (1, 1), (1, 1), (0, 0), 1]], 4, True], [
                      [[64, 128, (1, 1), (1, 1), (0, 0), 1], [128, 128, (3, 3), (1, 1), (1, 1), 128],
                       [128, 64, (1, 1), (1, 1), (0, 0), 1]], 4, True], [64, 24, (3, 3), (1, 1), (1, 1), 1],
                  [24, 24, (4, 4), (2, 2), (1, 1), 0, 24], [], [
                      [[56, 176, (1, 1), (1, 1), (0, 0), 1], [176, 176, (3, 3), (1, 1), (1, 1), 176],
                       [176, 56, (1, 1), (1, 1), (0, 0), 1]], 2, False], [56, 16, (3, 3), (2, 2), (1, 1), 1], [], [
                      [[80, 96, (1, 1), (1, 1), (0, 0), 1], [96, 96, (3, 3), (1, 1), (1, 1), 96],
                       [96, 48, (1, 1), (1, 1), (0, 0), 1]], 2, False], [
                      [[56, 64, (1, 1), (1, 1), (0, 0), 1], [64, 64, (3, 3), (1, 1), (1, 1), 64],
                       [64, 104, (1, 1), (1, 1), (0, 0), 1]], 2, False], [104, 16, (3, 3), (1, 1), (1, 1), 1],
                  [104, 24, (3, 3), (1, 1), (1, 1), 1], [
                      [[48, 112, (1, 1), (1, 1), (0, 0), 1], [112, 112, (3, 3), (1, 1), (1, 1), 112],
                       [112, 144, (1, 1), (1, 1), (0, 0), 1]], 2, False], [144, 16, (3, 3), (1, 1), (1, 1), 1],
                  [144, 24, (3, 3), (1, 1), (1, 1), 1]]
    net = SSD_Prune(net_params, block_cfg=MobileNetV2_4_2_prune, num_classes=num_classes)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.eval()

    print('{} model, anchors, and classes loaded.'.format(model_path))

    from pth2caffe import pytorch_to_caffe

    name = 'mobilenetv1'
    dummy_input = torch.ones([1, 3, 1024, 1024])    # 转换caffe时需要把输入图片尺寸设置为[1, 3, 1024, 1024]，在prototxt文件中在修改成[1, 3, 512, 512]
    pytorch_to_caffe.trans_net(net, dummy_input, name)
    pytorch_to_caffe.save_prototxt(param_path)
    pytorch_to_caffe.save_caffemodel(weight_path)


def test_caffemodel():
    # imgpath = "/media/z590/D/DataSet/BYD_obstacle/test/0823_record0/record_3/6.jpg"
    imgpath = "/media/z590/D/DataSet/BYD_corner_point/test/04-19_ps1_2022.2.24_CZ-T-N3_4.jpg"
    root = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/loss_2023_01_17_09_48_48/'
    param_path = root + 'mobilenetv2_4.prototxt'
    weight_path = root + 'mobilenetv2_4.caffemodel'

    input_shape = [256, 256]
    backbone = "corner_point_mobilenetv2_4"
    letterbox_image = True
    classes_path = 'data/voc_coner_point_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    # class_names = ['person','non','cone','pole','close','open']
    num_classes = num_classes + 1
    anchors = torch.from_numpy(get_anchors(input_shape, backbone)).type(torch.FloatTensor)


    cv_net = cv2.dnn.readNet(param_path, weight_path)
    srcimg = cv2.imread(imgpath)
    srcimg = cv2.resize(srcimg, input_shape, interpolation=cv2.INTER_AREA)
    image_shape = [srcimg.shape[0], srcimg.shape[1]]
    # srcimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)

    mean = np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, 3)
    # img = (img - mean)

    blob = cv2.dnn.blobFromImage(srcimg)
    # Sets the input to the network
    cv_net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outputs = cv_net.forward(cv_net.getUnconnectedOutLayersNames())
    outputs = list(outputs)
    bbox_util = BBoxUtility(num_classes)
    outputs[0] = torch.tensor(outputs[0])
    outputs[1] = torch.tensor(outputs[1])
    results = bbox_util.decode_box(outputs, anchors, image_shape, input_shape, letterbox_image,
                                                        nms_iou=0.3, confidence=0.1)
    # targets = []
    # bbox_util.get_ap(results, targets)
    if len(results[0]) > 0:
        top_label = np.array(results[0][:, 4], dtype='int32')
        top_conf = results[0][:, 5]
        top_boxes = results[0][:, :4]
        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            score = top_conf[i]
            box = top_boxes[i]
            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(srcimg.shape[0], np.floor(bottom).astype('int32'))
            right = min(srcimg.shape[1], np.floor(right).astype('int32'))

            label = '{} {:.4f}'.format(predicted_class, score)
            print(label, top, left, bottom, right)
            cv2.rectangle(srcimg, (left, top), (right, bottom), (0, 0, 255), 3)
            cv2.putText(srcimg, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)

    cv2.imshow('result', srcimg)
    cv2.waitKey()


def test_SSDcaffemodel():
    input_shape = [256, 256]
    classes_path = 'data/voc_coner_point_classes.txt'
    class_names, num_classes = get_classes(classes_path)

    root = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/loss_2023_01_30_17_45_50_MobileNetV2_4_2_finetune/'
    param_path = root + 'MobileNetV2_4_2_prune_finetune.prototxt'
    weight_path = root + 'MobileNetV2_4_2_prune_finetune.caffemodel'
    cv_net = cv2.dnn.readNet(param_path, weight_path)
    dir_path = '/media/z590/D/DataSet/BYD_corner_point/test/'
    exts = ['jpg']

    file_list = get_files(dir_path, exts)
    for i, file in enumerate(file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_list)))
        sys.stdout.flush()
        # file = imgpath
        path, name = os.path.split(file)
        srcimg = cv2.imread(file)
        # srcimg[:, :, 0] = srcimg[:, :, 0]*0.9
        img = cv2.resize(srcimg, input_shape, interpolation=cv2.INTER_AREA)

        # mean = np.array([128, 128, 128], dtype=np.float32).reshape(1, 1, 3)
        # img = (img - mean)

        blob = cv2.dnn.blobFromImage(img)
        # Sets the input to the network
        cv_net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        print(cv_net.getUnconnectedOutLayersNames())
        # out = cv_net.forward('relu1')
        outputs = cv_net.forward(cv_net.getUnconnectedOutLayersNames())
        results = list(outputs[0][0])
        for result in results[0]:
            label = np.array(result[1], dtype='int32')
            if label == 0:
                continue
            classid = int(label)
            predicted_class = class_names[int(label)-1]
            conf = result[2]
            left, top, right, bottom = result[3:]
            top, left, bottom, right = max(0, int(top*srcimg.shape[0])), max(0, int(left*srcimg.shape[1])), min(srcimg.shape[0], int(bottom*srcimg.shape[0])), min(srcimg.shape[1], int(right*srcimg.shape[1]))
            label = '{} {:.4f}'.format(predicted_class, conf)
            print(label, top, left, bottom, right)
            cv2.rectangle(srcimg, (left, top), (right, bottom), (0, 0, 255), 3)
            cv2.putText(srcimg, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        # cv2.imwrite(path+name.replace('.jpg', '_ results.png'), srcimg)
        cv2.imshow('result', srcimg)
        cv2.waitKey()


def calibrate(labels1, boxes1, labels2, boxes2, img, color):
    is_save = False
    for i, label in list(enumerate(labels1)):
        box0 = boxes1[i]
        flag = False
        for j, box1 in enumerate(boxes2):
            val = iou(box0, box1)
            if val > 0.6:# and labels2[j] == label:
                flag = True
        if not flag:
            is_save = True
            left = max(0, np.floor(box0[0]).astype('int32'))
            top = max(0, np.floor(box0[1]).astype('int32'))
            right = min(img.shape[1], np.floor(box0[2]).astype('int32'))
            bottom = min(img.shape[0], np.floor(box0[3]).astype('int32'))
            # label = '{}'.format(label)
            # print(label, top, left, bottom, right)
            cv2.rectangle(img, (left, top), (right, bottom), color, 3)
            cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
            # json_labels.append(predicted_class)
            # json_boxes.append([int(0.5 * (left + right)), int(0.5 * (top + bottom))])
            # json_types.append('point')
    return is_save


def test_txt():
    save_path = '/media/z590/D/DataSet/BYD_corner_point/train_dataSet/modify_label/'

    root = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/loss_2023_01_14_18_15_47_corner_point_mobilenetv2_4/'
    param_path = root + 'corner_point_mobilenetv2_4.prototxt'
    weight_path = root + 'corner_point_mobilenetv2_4.caffemodel'

    input_shape = [256, 256]
    classes_path = 'data/voc_coner_point_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    cv_net = cv2.dnn.readNet(param_path, weight_path)
    val_annotation_path     = '2012_val_coner_point.txt'
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
        for val_line in val_lines:
            line = val_line.split()
            file = line[0]
            path, name = os.path.split(file)
            if not os.path.exists('/media/z590/D/DataSet/BYD_corner_point/finish_label/20230112/Json/'+name):
                continue
            img = cv2.imread(file)#Image.open(line[0])
            srcimg = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            blob = cv2.dnn.blobFromImage(srcimg)
            # Sets the input to the network
            cv_net.setInput(blob)
            # Runs the forward pass to get output of the output layers
            outputs = cv_net.forward(cv_net.getUnconnectedOutLayersNames())
            results = list(outputs[0][0])

            if len(results[0]) > 0:
                json_labels, json_boxes, json_types = [], [], []
                labels, boxes = read_VOCxml(file.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml'))
                for i, box in enumerate(boxes):
                    json_labels.append(labels[i])
                    json_boxes.append([int(0.5*(box[0]+box[2])), int(0.5*(box[1]+box[3]))])
                    json_types.append('point')
                #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
                #     cv2.putText(img, labels[i], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
                top_labels = []
                pre_label = np.array(results[0][:, 1], dtype='int32')
                pre_label = np.delete(pre_label, np.where(pre_label == 0))
                for label in pre_label:
                    top_labels.append(class_names[int(label)-1])

                top_conf = results[0][:, 2]
                top_boxes = results[0][:, 3:]
                top_boxes[:, 0] = top_boxes[:, 0] * img.shape[1]
                top_boxes[:, 1] = top_boxes[:, 1] * img.shape[0]
                top_boxes[:, 2] = top_boxes[:, 2] * img.shape[1]
                top_boxes[:, 3] = top_boxes[:, 3] * img.shape[0]
                #calibrate(top_labels, top_boxes, labels, boxes, img, (0,0,255)) or
                if calibrate(labels, boxes, top_labels, top_boxes, img, (255,0,0)):
                    # shutil.copy(file, save_path+name)
                    cv2.imwrite(save_path+name, img)
                    shutil.copyfile(file.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml'), save_path+name.replace('.jpg', '.xml'))
                    # creat_json(save_path, name, json_labels, json_boxes, json_types, 512, 512)

            # cv2.imshow('result', img)
            # cv2.waitKey()


def pth2onnx():
    import onnx
    import onnxruntime as ort
    import onnxsim
    do_simplify = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/loss_2022_11_08_19_47_18_JacintoNetV2_yihang/'
    # root = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/'
    model_path = root + 'ep100-loss1.401-val_loss2.358.pth'
    onnx_path = root + 'test_onnx.onnx'

    input_shape = [256, 256]
    backbone = "JacintoNetV2_yihang"
    letterbox_image = True
    classes_path = 'data/voc_coner_point_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    # class_names = ['person','non','cone','pole','close','open']
    num_classes = num_classes + 1
    anchors = torch.from_numpy(get_anchors(input_shape, backbone)).type(torch.FloatTensor)

    net = SSD300(num_classes, backbone, False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.eval()

    # Input
    inputs = torch.randn(1, 3, 256, 256)
    torch.onnx.export(net,
                      inputs,
                      onnx_path,
                      verbose=True,
                      do_constant_folding=False,
                      opset_version=9,
                      input_names=['data'],
                      output_names=['seg_out'])
    print('convert', onnx_path, 'to onnx finish!!!')

    model_onnx = onnx.load(onnx_path)  # load onnx model
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


if __name__ == "__main__":
    # pth2caffe2()
    # pth2onnx()
    # test_txt()
    # test_caffemodel()
    test_SSDcaffemodel()





