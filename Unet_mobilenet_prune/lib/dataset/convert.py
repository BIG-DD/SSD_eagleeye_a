import xml.etree.ElementTree as ET
import json
import numpy as np
import shutil
import os


# bdd_labels = {
# 'unlabeled':0, 'dynamic': 1, 'ego vehicle': 2, 'ground': 3, 
# 'static': 4, 'parking': 5, 'rail track': 6, 'road': 7, 
# 'sidewalk': 8, 'bridge': 9, 'building': 10, 'fence': 11, 
# 'garage': 12, 'guard rail': 13, 'tunnel': 14, 'wall': 15,
# 'banner': 16, 'billboard': 17, 'lane divider': 18,'parking sign': 19, 
# 'pole': 20, 'polegroup': 21, 'street light': 22, 'traffic cone': 23,
# 'traffic device': 24, 'traffic light': 25, 'traffic sign': 26, 'traffic sign frame': 27,
# 'terrain': 28, 'vegetation': 29, 'sky': 30, 'person': 31,
# 'rider': 32, 'bicycle': 33, 'bus': 34, 'car': 35, 
# 'caravan': 36, 'motorcycle': 37, 'trailer': 38, 'train': 39,
# 'truck': 40       
# }
id_dict = {'person': 0, 'rider': 1, 'car': 2, 'bus': 3, 'truck': 4,
'bike': 5, 'motor': 6, 'tl_green': 7, 'tl_red': 8,
'tl_yellow': 9, 'tl_none': 10, 'traffic sign': 11, 'train': 12}
#
# id_dict = {'horizon': 0, 'vertical': 1, 'incline': 2, 'occupancy': 3}

id_dict_single = {'car': 0, 'bus': 1, 'truck': 2,'train': 3}
# id_dict = {'car': 0, 'bus': 1, 'truck': 2}

voc_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
#
parking_slot = {'horizon': 0, 'vertical': 1, 'incline': 2, 'horizon_vehicle': 3, 'vertical_vehicle': 4,
                'incline_vehicle': 5, 'lock': 6, 'unlock': 7, 'obstacle': 8, 'no_parking': 9, 'people': 10,
                'limit_rod': 11}

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    center_x = (box[0] + box[1])/2.0
    center_y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    center_x = center_x*dw
    w = w*dw
    center_y = center_y*dh
    h = h*dh
    return (center_x, center_y, w, h)


def read_VOCxml(file_path, width, height):
    tree = ET.ElementTree(file=file_path)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    # file_name = root.findall('filename')
    # file_name = file_name[0].text
    # size_set = root.findall('size')
    # width = size_set[0].find('width').text
    # height = size_set[0].find('height').text

    object_set = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    gt = np.zeros((len(object_set), 5))
    for idx, object in enumerate(object_set):
        obj_label = object.find('name').text
        bnd_box = object.find('bndbox')
        x1 = int(float(bnd_box.find('xmin').text))  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(float(bnd_box.find('ymin').text))  # -1
        x2 = int(float(bnd_box.find('xmax').text))  # -1
        y2 = int(float(bnd_box.find('ymax').text))  # -1
        if obj_label.isdigit():
            cls_id = obj_label
        else:
            cls_id = voc_dict[obj_label]
        # if single_cls:
        #     cls_id = 0
        gt[idx][0] = cls_id  # obj_label#
        box = convert((width, height), (x1, x2, y1, y2))
        gt[idx][1:] = list(box)

    return gt


def read_json(label_path, width, height):
    with open(label_path, 'r', encoding='utf-8') as f:
        setting = json.load(f)
    coordinate = setting['shapes']
    count = 0
    for j in range(len(coordinate)):
        label = coordinate[j]['label']
        if label == 'lane':
            count += 1

    gt = np.zeros((len(coordinate)-count, 5))
    indx = 0
    for j in range(len(coordinate)):
        label = coordinate[j]['label']
        if label == 'lane':
            continue
        cls_id = parking_slot[label]
        gt[indx][0] = cls_id

        point = coordinate[j]['points']
        if len(point) != 2:
            print(label_path+' point error!')

        shape_type = coordinate[j]['shape_type']
        if shape_type != 'rectangle' and label in ['horizon', 'vertical', 'incline'] :
            min_x, min_y, max_x, max_y = 1000, 1000, 0, 0
            for p in point:
                min_x = min(min_x, p[0])
                min_y = min(min_y, p[1])
                max_x = max(max_x, p[0])
                max_y = max(max_y, p[1])
            point = [[min_x, min_y], [max_x, max_y]]

        box = convert((width, height), (point[0][0], point[1][0], point[0][1], point[1][1]))
        gt[indx][1:] = list(box)
        indx += 1

    return gt


def filter_data(data):
    remain = []
    for obj in data:
        if 'box2d' in obj.keys():  # obj.has_key('box2d'):
            remain.append(obj)
    return remain


