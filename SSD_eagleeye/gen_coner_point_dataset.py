import os, sys
from writeVOCxml import GEN_Annotations
import json
import numpy as np
import cv2
import copy
import base64
import math
import shutil
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET


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


def read_json(file):
    points, labels, shape_types, boxes = [], [], [], []
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
        point = point[0]
        xmin = int(max(0, point[0] - 20))
        ymin = int(max(0, point[1] - 20))
        xmax = int(min(512, point[0] + 20))
        ymax = int(min(512, point[1] + 20) )
        boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

    return labels, points, boxes, shape_types


def creat_json(save_path, name, labels, points, shape_types, h, w):
    with open(save_path+name.replace('.json', '.jpg').replace('.xml', '.jpg'), 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    shapes_context = []
    for i, point in enumerate(points):
        str_points = []
        for coor in point:
            str_points.append([(math.ceil(coor[0])), (math.ceil(coor[1]))])
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

    with open(save_path + '/' + name.replace('.jpg', '.json').replace('.xml', '.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(json_str, indent=2, ensure_ascii=False))


def read_VOCxml(file_path):
    labels, boxes = [], []
    tree = ET.ElementTree(file=file_path)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    object_set = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    for object in object_set:
        obj_name = object.find('name').text
        bnd_box = object.find('bndbox')
        x1 = int(float(bnd_box.find('xmin').text))  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(float(bnd_box.find('ymin').text))  # -1
        x2 = int(float(bnd_box.find('xmax').text))  # -1
        y2 = int(float(bnd_box.find('ymax').text))  # -1
        # if 'traffic_cone' != obj_name:
        boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        labels.append(obj_name)

    return labels, boxes


def creat_xml(save_path, name, labels, boxes, W, H):
    anno = GEN_Annotations(name.replace('.json', '.jpg'))
    anno.set_size(W, H, 3)
    for i in range(len(boxes)):
        box = boxes[i]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])-x
        h = int(box[3])-y
        label = labels[i]
        anno.add_pic_attr(label, x, y, w, h)
    anno.savefile(save_path + name.replace('.jpg', '.xml'))


def rotateCordiate(coor, origin_size, new_size, radian):
    new_coor = []
    for point in coor:
        points = []
        # for point in pts:
        x = (point[0] - math.ceil(origin_size[0] / 2)) * math.cos(radian) + (
                point[1] - math.ceil(origin_size[1] / 2)) * math.sin(radian) + math.ceil(new_size[0] / 2)
        y = -(point[0] - math.ceil(origin_size[0] / 2)) * math.sin(radian) + (
                point[1] - math.ceil(origin_size[1] / 2)) * math.cos(radian) + math.ceil(new_size[1] / 2)
        points.append(np.array([int(x), int(y)]))
        new_coor.append(np.array([int(x), int(y)]))
    return new_coor


def rotateImageTheterBoxes(im, theter, coor):
    ori_h, ori_w = im.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((ori_w / 2, ori_h / 2), theter, 1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((ori_h * sin) + (ori_w * cos))
    nH = int((ori_h * cos) + (ori_w * sin))

    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (nW / 2) - ori_w / 2
    rotation_matrix[1, 2] += (nH / 2) - ori_h / 2

    im_rotate = cv2.warpAffine(im, rotation_matrix, (nW, nH), borderValue=(0, 0, 0))

    new_h, new_w = im_rotate.shape[:2]
    new_size = [new_w, new_h]

    # 旋转前的坐标对应的旋转后的坐标
    radian = theter * 3.1415926 / 180.0
    ori_h, ori_w = im.shape[:2]
    origin_size = [ori_w, ori_h]

    new_coor = rotateCordiate(coor, origin_size, new_size, radian)

    return im_rotate, new_coor


def json2xml():
    root = '/media/z590/D/DataSet/BYD_corner_point/train_dataSet/20230117/'
    dir_path = '/media/z590/D/DataSet/BYD_corner_point/finish_label/20230112/Json'
    save_anno = root + 'VOC2012/Annotations/'
    save_img = root + 'VOC2012/JPEGImages/'
    img_size = [512, 512]
    if not os.path.exists(save_anno):
        os.makedirs(save_anno)
    if not os.path.exists(save_img):
        os.makedirs(save_img)
    exts = ['json']
    file_list = get_files(dir_path, exts)
    classes = []
    for i, file in enumerate(file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_list)))
        sys.stdout.flush()
        _, name = os.path.split(file)
        xml_boxes, xml_labels, xml_points = [], [], []
        labels, points, boxes, shape_types = read_json(file.replace('.jpg', '.json'))

        img = cv2.imdecode(np.fromfile(file.replace('.json', '.jpg'), dtype=np.uint8), -1)
        h, w, _ = img.shape
        rate_x = 1.0*img_size[0]/w
        rate_y = 1.0*img_size[1]/h
        img = cv2.resize(img, (img_size))
        for j, box in enumerate(points):
            if shape_types[j] == 'point':
                box = box[0]
                xml_labels.append(str.upper(labels[j]))
                xmin = int(max(0, box[0]-20)*rate_x)
                ymin = int(max(0, box[1]-20)*rate_y)
                xmax = int(min(w, box[0]+20)*rate_x)
                ymax = int(min(h, box[1]+20)*rate_y)
                xml_boxes.append([xmin, ymin, xmax, ymax])
                xml_points.append(box)
            if labels[j] not in classes:
                classes.append(labels[j])

        # shutil.copy(file.replace('.json', '.jpg'), )
        cv2.imwrite(save_img + name.replace('.json', '.jpg'), img)
        creat_xml(save_anno, name.replace('.json', '.jpg'), xml_labels, xml_boxes, img_size[0], img_size[1])

        # augment rotate
        aug_times = 0
        while (aug_times):
            aug_times -= 1
            theter = np.random.randint(0, 180)
            im_rotate, new_coor = rotateImageTheterBoxes(copy.deepcopy(img), theter, copy.deepcopy(xml_points))
            W, H, _ = im_rotate.shape
            pad_w, pad_h = int(0.5*(W-img_size[1])), int(0.5*(H-img_size[0]))
            new_img = im_rotate[pad_h:pad_h+img_size[0], pad_w:pad_w+img_size[1], :]
            new_labels, new_boxes = [], []
            for j, box in enumerate(new_coor):
                xmin = int(max(0, box[0] - 20))-pad_w
                ymin = int(max(0, box[1] - 20))-pad_h
                xmax = int(min(W, box[0] + 20))-pad_w
                ymax = int(min(H, box[1] + 20))-pad_h
                if not (xmin<20 or ymin<20 or xmax>img_size[1] or ymax>img_size[0]):
                    new_labels.append(xml_labels[j])
                    new_boxes.append([xmin, ymin, xmax, ymax])
            new_name = name.replace('.json', '_'+str(aug_times)+'.json')
            cv2.imwrite(save_img + new_name.replace('.json', '.jpg'), new_img)
            creat_xml(save_anno, new_name.replace('.json', '.jpg'), new_labels, new_boxes, img_size[0], img_size[1])

    print(classes)


def move_file():
    dir_path = '/media/z590/D/DataSet/BYD_corner_point/finish_label/corner_points/'
    save_path = '/media/z590/D/DataSet/BYD_corner_point/finish_label/error/'
    exts = ['jpg']
    file_list = get_files(dir_path, exts)
    for i, file in enumerate(file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_list)))
        sys.stdout.flush()
        _, name = os.path.split(file)
        if os.path.exists('/media/z590/D/DataSet/BYD_corner_point/finish_label/finish/part2/'+name):
            labels, boxes, shape_types = read_json(file.replace('.jpg', '.json'))
            flag = False
            for label in labels:
                if label not in ['I', 'T', 'Y', 'U', 'L']:
                    flag = True
            if flag:
                shutil.copy('/media/z590/D/DataSet/BYD_corner_point/finish_label/finish/part2/'+name, save_path+name)
                shutil.copy('/media/z590/D/DataSet/BYD_corner_point/finish_label/finish/part2/'+name.replace('.jpg', '.json'), save_path+name.replace('.jpg', '.json'))


def xml2json():
    dir_path = '/media/z590/D/DataSet/BYD_corner_point/finish_label/20230112/Annotations/'
    save_path = '/media/z590/D/DataSet/BYD_corner_point/finish_label/20230112/Json/'
    exts = ['xml']
    file_list = get_files(dir_path, exts)
    for i, file in enumerate(file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_list)))
        sys.stdout.flush()
        _, name = os.path.split(file)
        img_file = file.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg')
        img = cv2.imread(img_file)
        W, H, _ = img.shape
        if W != 512 or H != 512:
            print()
        rate_x, rate_y = 512.0/W, 512.0/H
        json_labels, json_points, json_shape_types = [], [], []
        labels, boxes = read_VOCxml(file)
        for j, box in enumerate(boxes):
            w, h = box[2]-box[0], box[3]-box[1]
            center_x, center_y = 0.5*(box[2]+box[0]), 0.5*(box[3]+box[1])
            if w != 40:
                if box[0] == 0:
                    center_x = box[2]-20
                else:
                    center_x = box[0]+20
            if h != 40:
                if box[1] == 0:
                    center_y = box[3]-20
                else:
                    center_y = box[1]+20
            center_x, center_y = int(center_x*rate_x), int(center_y*rate_y)
            json_labels.append(labels[j])
            json_points.append([[center_x, center_y]])
            json_shape_types.append('point')
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(save_path+name.replace('.xml', '.jpg'), img)
        creat_json(save_path, name, json_labels, json_points, json_shape_types, 512, 512)

        # print(boxes)


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


def calibrate(labels1, boxes1, labels2, boxes2):
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
    return is_save


def jsonComparejson():
    save_path = '/media/z590/D/DataSet/BYD_corner_point/finish_label/20230112/part12_jpg_json_modify/'
    save_path1 = '/media/z590/D/DataSet/BYD_corner_point/finish_label/20230112/no/'
    dir_path = '/media/z590/D/DataSet/BYD_corner_point/finish_label/20230112/part12_jpg_json'
    exts = ['json']
    file_list = get_files(dir_path, exts)
    for i, file in enumerate(file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_list)))
        sys.stdout.flush()
        _, name = os.path.split(file)
        labels0, _, boxes0, shape_types0 = read_json(file.replace('.jpg', '.json'))
        if not os.path.exists(file.replace('part12_jpg_json', 'Json')):
            shutil.copyfile(file.replace('.jpg', '.json'), save_path1 + name)
            shutil.copyfile(file.replace('.json', '.jpg'), save_path1 + name.replace('.json', '.jpg'))
            continue
        labels1, _, boxes1, shape_types1 = read_json(file.replace('part12_jpg_json', 'Json').replace('.jpg', '.json'))
        if calibrate(labels0, boxes0, labels1, boxes1) or calibrate(labels1, boxes1, labels0, boxes0):
            shutil.copyfile(file.replace('.jpg', '.json'), save_path+name)
            shutil.copyfile(file.replace('.json', '.jpg'), save_path+name.replace('.json', '.jpg'))


def json2json():
    save_path = '/media/z590/D/DataSet/BYD_corner_point/finish_label/error_done_modify/'
    dir_path = '/media/z590/D/DataSet/BYD_corner_point/finish_label/error_done/'
    exts = ['json']
    file_list = get_files(dir_path, exts)
    for i, file in enumerate(file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_list)))
        sys.stdout.flush()
        _, name = os.path.split(file)
        json_labels, json_points, json_shape_types = [], [], []
        labels, points, boxes, shape_types = read_json(file.replace('.jpg', '.json'))

        img = cv2.imdecode(np.fromfile(file.replace('.json', '.jpg'), dtype=np.uint8), -1)
        h, w, _ = img.shape
        rate_x = 512.0 / w
        rate_y = 512.0 / h
        if rate_x == 1 and rate_y == 1:
            continue
        img = cv2.resize(img, (512, 512))
        for j, shape_type in enumerate(shape_types):
            point = points[j]
            for k, p in enumerate(point):
                points[j][k][0], points[j][k][1] = [p[0]*rate_x, p[1]*rate_y]

        cv2.imwrite(save_path + name.replace('.json', '.jpg'), img)
        creat_json(save_path, name, labels, points, shape_types, 512, 512)


if __name__ == '__main__':
    json2xml()



