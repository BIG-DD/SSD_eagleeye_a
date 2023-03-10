import json
import base64
import math
import copy
import shutil
import cv2
import os
import numpy as np


def read_json(file):
    points = []
    labels = []
    shape_types = []
    with open(file, 'r', encoding='utf-8') as f:
        setting = json.load(f)
    coordinate = setting['shapes']

    for j in range(len(coordinate)):
        label = coordinate[j]['label']
        point = coordinate[j]['points']
        shape_type = coordinate[j]['shape_type']
        labels.append(label)
        points.append(point)
        shape_types.append(shape_type)

    return labels, points, shape_types

def creat_json(dir_path, name, labels, points, shape_types, h, w):
    with open(dir_path+name.split('.j')[0] + '.jpg', 'rb') as f:
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

    with open(dir_path + '/' + name.split('.')[0] + '.json', "w", encoding='utf-8') as f:
        f.write(json.dumps(json_str, indent=2, ensure_ascii=False))
# json文件路径
json_dir_all = '/media/byd/D/data/ps2_parkingslot_check_done'
# json保存路径
save_dir_all = '/media/byd/D/data/public_lane_data_ps2/ps2_check_done'
# 有红框的图片路径
file_dir = '/media/byd/D/data/public_lane_data_ps2/jpg'

two_file_dir_list = os.listdir(json_dir_all)#json_name,..
json_file_list = []
for file_name in two_file_dir_list:
    if file_name.split('.j')[-1]=='son':
        json_file_list.append(file_name)

for indx, img_name in enumerate(json_file_list):
    print(indx)
    json_one_dir = json_dir_all + '/' + img_name
    with open(json_one_dir, 'r', encoding='utf-8') as f:
        sett = json.load(f)
    #img_hight = sett['imageHeight']
    #img_width = sett['imageWidth']
    base64_str = sett['imageData']
    # byte_data = base64.b64decode(base64_str)
    # encode_image = np.asarray(bytearray(byte_data), dtype="uint8")  #
    # img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)  #
    img_hight = sett['imageHeight']
    img_width = sett['imageWidth']
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # BGR2RGB
    # cv2.imwrite('1.jpg', img_array)
    img_path = file_dir + '/' + img_name.replace('.json', '.jpg')
    img_1 = cv2.imread(img_path)
    img_2 = cv2.resize(img_1, (int(img_width), int(img_hight)))
    cv2.imwrite(save_dir_all + '/' + img_name.replace('.json', '.jpg'), img_2)
    seet1 = copy.deepcopy(sett)
    with open(save_dir_all + '/' +img_name.replace('.json', '.jpg'), 'rb') as g:
        img_data = g.read()
        imageData = base64.b64encode(img_data).decode('utf-8')
    seet1['imageData'] = imageData
    with open(save_dir_all + '/' +img_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(seet1, indent=2, ensure_ascii=False))
    print('write json .....', save_dir_all + '/' +img_name)



# with open(json_dir, 'r', encoding='utf-8') as f:
#     sett = json.load(f)
# img_hight = sett['imageHeight']
# img_width = sett['imageWidth']
#
# seet1 = copy.deepcopy(sett)
# with open(img_dir, 'rb') as g:
#     img_data = g.read()
#     imageData = base64.b64encode(img_data).decode('utf-8')
# seet1['imageData'] = imageData
# with open('04-19_ps1_2022.2.24_CZ-T-N3_20.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(seet1, indent=2, ensure_ascii=False))