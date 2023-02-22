import base64
import copy
import json
import numpy as np
import cv2
import os
import sys
import math
import random
import shutil
import argparse
from random import choice
from tkinter import _flatten
from PIL import Image
from datetime import datetime


PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
bin_colormap = list(_flatten(PALETTE))


def parse_args():
    parser = argparse.ArgumentParser(description='Generated data')
    parser.add_argument('--rotate',
                        default=True,
                        type=bool,
                        help='Image rotation enhancement')
    parser.add_argument('--crop',
                        default=True,
                        type=bool,
                        help='Image rotation enhancement')
    parser.add_argument('--times',
                        type=int,
                        default=2,
                        help='Data increment times')
    parser.add_argument('--vertical_transform_incline',
                        type=bool,
                        default=True,
                        help='Image vertical transform incline')
    parser.add_argument('--image_json_dir',
                        type=str,
                        default='/media/adas3/data/corner/data/all_data_finish_time1/VOC2012/JPEGImages/',
                        help='Image and json dir for make mask')
    parser.add_argument('--origin_json_dir',
                        type=str,
                        default='/media/adas3/data/corner/data/all_data_finish_time1/VOC2012/origin_json/',
                        help='get vertical label')
    parser.add_argument('--save_dir',
                        type=str,
                        default='/media/adas3/data/corner/data/all_data_finish_time1/VOC2012/image_json/',
                        help='data directory')
    parser.add_argument('--is_show',
                        type=bool,
                        default=True,
                        help='save label image')
    args = parser.parse_args()

    return args

def main(aug_mask=False):
    arg = parse_args()
    save_path = arg.save_dir
    data_path = arg.image_json_dir
    times = arg.times
    origin_json = arg.origin_json_dir

    exts = ['jpg']
    file_lists = get_files(data_path, exts)
    random.shuffle(file_lists)
    for i, file in enumerate(file_lists):
        sys.stdout.write('\r>> Converting images %d/%d' % (i + 1, len(file_lists)))
        sys.stdout.flush()
        path, name = os.path.split(file)
        print(name)
        # if name != '04-22_3_ps63_2022.4.14_CZ-N6_183.jpg':
        #     continue
        # if 'CBPSD' not in name:
        #     continue
        if i % 8 == 0:
            train_val = 'val/'
        else:
            train_val = 'train/'

        if arg.is_show and i % 100 == 0:
            is_show = True
        else:
            is_show = False
        # if os.path.exists(file.replace('.jpg', '.json')):
        #     read_json(file.replace('.jpg', '.json'))

        json_dir = origin_json + name.replace('.jpg', '.json')
        if os.path.exists(json_dir):
            if arg.vertical_transform_incline:
                vertical_transform_incline(json_dir, file, save_path, train_val, times, rotate=arg.rotate, crop=arg.crop, is_show=is_show, aug_mask=aug_mask)

        # create_dataset(file, save_path, train_val, times, rotate=arg.rotate, crop=arg.crop, is_show=is_show, aug_mask=aug_mask)


def vertical_transform_incline(json_file, mask_file, save_path, train_val_tr, times, rotate, crop, is_show, aug_mask,
                               resize_w=512, resize_h=512, angles=[-35, -30, -25, 25, 30, 35]):
    temp_times = times
    vertical_points = []
    is_vertical = False
    is_horizon = False
    path, name = os.path.split(json_file)#原始文件
    if aug_mask:
        temp_times = 3 * times
        name = name.replace('.json', '_aug_mask.json')

    labels, points, shape_types = read_json(json_file)
    # 获取透视变换坐标，完成透视变换
    # 这部分暂时没用
    '''
    for j, point in enumerate(points):
        if labels[j] == 'T' or labels[j] == 'L':
            print(len(point))
            print(int(point[0][0]), int(point[0][1]))
            if len(point) == 1:
                vertical_points.append([int(point[0][0]), int(point[0][1])])
                vertical_points.append([int(point[1][0]), int(point[1][1])])
                if abs(point[0][0] - point[1][0]) < 5:
                    is_vertical = True
                elif abs(point[0][1] - point[1][1]) < 5:
                    is_horizon = True
    '''
    for j, point in enumerate(points):
        if labels[j] == 'T' or labels[j] == 'L':
            vertical_points.append([int(point[0][0]), int(point[0][1])])
    is_vertical = True
    if is_horizon or is_vertical and not (is_horizon and is_vertical):
        img_path = str(json_file).replace('.json', '.jpg')
        img_path = mask_file # + json_file.split('/')[-1].replace('.json', '.jpg')
        ori_image = cv2.imread(img_path)
        height, width, _ = ori_image.shape

        image, gt_mask, is_polygon = get_aug_mask(mask_file, aug_mask)
        if is_polygon:
            train_val_tr = train_val_tr.replace('val/', 'train/')

        gt_mask = cv2.resize(gt_mask, (width, height), cv2.INTER_NEAREST)
        image = cv2.resize(image, (width, height))
        # if '05-06_3_ps33_2022.4.8_CZ-N6_128' in name:
        #     print()
        # 计算车位线角点，确定透视变换点
        if is_vertical:
            for i in range(temp_times):
                image1 = copy.deepcopy(image)
                gt_mask1 = copy.deepcopy(gt_mask)
                gt_mask1[gt_mask1 > 0] = 255
                left_min_y, left_max_y, left_mid_x = 1000, 0, []
                right_min_y, right_max_y, right_mid_x = 1000, 0, []
                for point in vertical_points:
                    if point[0] < 0.5 * width:
                        left_mid_x.append(point[0])
                        left_min_y = min(left_min_y, point[1])
                        left_max_y = max(left_max_y, point[1])
                    else:
                        right_mid_x.append(point[0])
                        right_min_y = min(right_min_y, point[1])
                        right_max_y = max(right_max_y, point[1])
                is_save = False
                is_right, is_left = False, False
                if len(left_mid_x) > 3:
                    std_x = np.std(left_mid_x)
                    if std_x < 10:
                        is_save = True
                        is_left = True
                        left_mid_x = int(np.mean(left_mid_x))
                        src = np.array([[left_mid_x, left_min_y], [left_mid_x, left_max_y], [0, left_max_y], [0, left_min_y]],
                                       np.float32)
                        theter = choice(angles)
                        padding = (left_mid_x) * np.tan(theter * np.pi / 180)
                        dst = np.array([[left_mid_x, left_min_y], [left_mid_x, left_max_y], [0, left_max_y + padding],
                                        [0, left_min_y + padding]], np.float32)
                        crop_img = image[:, 0:left_mid_x, :]
                        result = transform(copy.deepcopy(crop_img), src, dst, height, left_mid_x)
                        # cv2.imshow("temp2", crop_img)
                        # cv2.imshow("temp3", result)
                        # cv2.waitKey(0)
                        try:
                            image1[:, 0:left_mid_x, :] = result
                        except:
                            is_save = False
                        # cv2.imwrite('11.jpg', result)

                        # crop_img = gt_mask1[:, 0:left_mid_x]
                        # result = transform(copy.deepcopy(crop_img), src, dst, height, left_mid_x)
                        # result[result > 0] = 255
                        # gt_mask1[:, 0:left_mid_x] = result

                        # cv2.imwrite('11.jpg', result)

                if len(right_mid_x) > 3:
                    std_x = np.std(right_mid_x)
                    if std_x < 10:
                        is_save = True
                        is_right = True
                        right_mid_x = int(np.mean(right_mid_x))
                        src = np.array([[0, right_min_y], [0, right_max_y], [width - right_mid_x, right_max_y],
                                        [width - right_mid_x, right_min_y]], np.float32)
                        theter = choice(angles)
                        padding = (width - right_mid_x) * np.tan(theter * np.pi / 180)

                        dst = np.array([[0, right_min_y], [0, right_max_y], [width - right_mid_x, right_max_y + padding],
                                        [width - right_mid_x, right_min_y + padding]], np.float32)
                        crop_img = image[:, right_mid_x:, :]
                        # print()
                        result = transform(copy.deepcopy(crop_img), src, dst, height, width - right_mid_x)
                        # cv2.imshow("temp", crop_img)
                        # cv2.imshow("temp1", result)
                        # cv2.waitKey(0)
                        try:
                            image1[:, right_mid_x:, :] = result
                        except:
                            is_save = False

                        # crop_img = gt_mask1[:, right_mid_x:]
                        # result = transform(copy.deepcopy(crop_img), src, dst, height, width - right_mid_x)
                        # result[result > 0] = 255
                        # gt_mask1[:, right_mid_x:] = result

                if is_save:
                    # image1 = cv2.resize(image1, (resize_w, resize_h))
                    # gt_mask1 = cv2.resize(gt_mask1, (resize_w, resize_h), cv2.INTER_NEAREST)

                    image_train_val_dir = save_path + 'images/'
                    ll_seg_annotations_dir = save_path + 'json/'

                    if not os.path.exists(image_train_val_dir):
                        os.makedirs(image_train_val_dir)
                    if not os.path.exists(ll_seg_annotations_dir):
                        os.makedirs(ll_seg_annotations_dir)

                    cv2.imwrite(image_train_val_dir + name.replace('.json', '.jpg'), image1)
                    # 生成标签 json
                    width, height, channels = image1.shape

                    # 定义通过Labelme标注后生成的json文件
                    with open(image_train_val_dir + name.replace('.json', '.jpg'), 'rb') as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode('utf-8')
                    shapes_context = []
                    for i, point in enumerate(points):
                        if labels[i] == 'T' or labels[i] == 'L':
                            if is_left and point[0][0] < 0.5 * width:
                                shape_context = {
                                    "shape_type": 'point',
                                    "label": 'Y',
                                    "line_color": None,
                                    "points": [[int(point[0][0]), int(point[0][1])]],
                                    "fill_color": None
                                }
                                shapes_context.append(shape_context)
                            elif is_left is False and point[0][0] < 0.5 * width:
                                shape_context = {
                                    "shape_type": 'point',
                                    "label": labels[i],
                                    "line_color": None,
                                    "points": [[int(point[0][0]), int(point[0][1])]],
                                    "fill_color": None
                                }
                                shapes_context.append(shape_context)
                            if is_right and point[0][0] > 0.5 * width:
                                shape_context = {
                                    "shape_type": 'point',
                                    "label": 'Y',
                                    "line_color": None,
                                    "points": [[int(point[0][0]), int(point[0][1])]],
                                    "fill_color": None
                                }
                                shapes_context.append(shape_context)
                            elif is_right is False and point[0][0] > 0.5 * width:
                                shape_context = {
                                    "shape_type": 'point',
                                    "label": labels[i],
                                    "line_color": None,
                                    "points": [[int(point[0][0]), int(point[0][1])]],
                                    "fill_color": None
                                }
                                shapes_context.append(shape_context)
                    json_str = {
                        "flags": {},
                        "imagePath": name,
                        "shapes": shapes_context,
                        "imageData": imageData,
                        "version": "4.6.0",
                        "lineColor": None,
                        "imageHeight": height,
                        "imageWidth": width,
                        "fillColor": None
                    }
                    with open(ll_seg_annotations_dir + name, "w", encoding='utf-8') as f:
                        f.write(json.dumps(json_str, indent=2, ensure_ascii=False))
                    # cv2.imwrite(ll_seg_annotations_dir +name.replace('.json', '_inc_'+str(i)+'.png'), gt_mask1)
                    # if is_show and i == 0:
                    #     save_path_show = save_path+'is_show/'
                    #     if not os.path.exists(save_path_show):
                    #         os.makedirs(save_path_show)
                    #     show_label(image1, gt_mask1, save_path_show, name.replace('.json', '_inc.jpg'))
                    #
                    # if rotate:
                    #     agument_rotate(temp_times, copy.deepcopy(image1), copy.deepcopy(gt_mask1), image_train_val_dir,
                    #                    ll_seg_annotations_dir, name.replace('.json', '.jpg'), resize_h, resize_w)
                    #
                    # if crop:
                    #     agument_crop(temp_times, copy.deepcopy(image1), copy.deepcopy(gt_mask1), image_train_val_dir,
                    #                  ll_seg_annotations_dir, name.replace('.json', '.jpg'), resize_h, resize_w)


def transform(img, src, dst, height, width):
    transform = cv2.getPerspectiveTransform(src, dst)
    # print(transform)
    result = cv2.warpPerspective(img, transform, (width, height))  # 透视变换
    # cv2.imwrite('result.jpg', result)
    # cv2.imwrite('img.jpg', img)

    return result


def get_files(files_path, exts):
    '''
    find images files in data path
    :return: list of files found
    '''
    files = []
    for parent, dirnames, filenames in os.walk(files_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
    return files


def rotateImageTheter(im, theter):
    ori_h, ori_w = im.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((ori_w / 2, ori_h / 2), theter, 1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    # compute the new bounding dimensions of the images
    nW = int((ori_h * sin) + (ori_w * cos))
    nH = int((ori_h * cos) + (ori_w * sin))

    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (nW / 2) - ori_w / 2
    rotation_matrix[1, 2] += (nH / 2) - ori_h / 2

    im_rotate = cv2.warpAffine(im, rotation_matrix, (nW, nH), borderValue=(0, 0, 0))

    return im_rotate


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


def get_mask(file):
    is_polygon = False
    image = cv2.imread(file)
    path, name = os.path.split(file)
    image_h, image_w, _ = image.shape
    label_img = np.zeros((image_h, image_w), dtype=np.uint8)
    thickness = int(5 * max(image_h, image_w)/512.0)
    if 'ps2' == name[:3]:
        thickness = int(6 * max(image_h, image_w) / 512.0)
    elif 'CBPSD' == name[:5]:
            thickness = int(4 * max(image_h, image_w) / 512.0)
    if not os.path.exists(file.replace('.jpg', '.json')):
        mask = label_img
    else:
        labels, points, shape_types = read_json(file.replace('.jpg', '.json'))
        for j, point in enumerate(points):
            if len(point) > 1 and shape_types[j] == 'line':
                for k in range(len(point) - 1):
                    cv2.line(label_img, (int(point[k][0]), int(point[k][1])),
                             (int(point[k + 1][0]), int(point[k + 1][1])),
                             255, thickness=thickness)

            elif shape_types[j] == 'polygon':
                point = np.array(point, dtype=int)
                cv2.drawContours(label_img, [point], -1, 255, thickness=-1)
                is_polygon = True
        label_img[label_img>0] = 255
        # _, binary = cv2.threshold(label_img, 50, 255, cv2.THRESH_BINARY)
        mask = label_img
    return image, mask, is_polygon


def get_aug_mask(file, is_aug_mask):
    is_polygon = False
    image = cv2.imread(file)
    path, name = os.path.split(file)
    image_h, image_w, _ = image.shape
    label_img = np.zeros((image_h, image_w), dtype=np.uint8)
    thickness = int(5 * max(image_h, image_w)/512.0)
    if 'ps2' == name[:3]:
        thickness = int(6 * max(image_h, image_w) / 512.0)
    elif 'CBPSD' == name[:5]:
            thickness = int(4 * max(image_h, image_w) / 512.0)
    if not os.path.exists(file.replace('.jpg', '.json')):
        mask = label_img
    else:
        labels, points, shape_types = read_json(file.replace('.jpg', '.json'))
        for j, point in enumerate(points):
            if len(point) > 1 and shape_types[j] == 'line':
                for k in range(len(point) - 1):
                    cv2.line(label_img, (int(point[k][0]), int(point[k][1])),
                             (int(point[k + 1][0]), int(point[k + 1][1])),
                             255, thickness=thickness)

            elif shape_types[j] == 'polygon':
                is_polygon = True
                point = np.array(point, dtype=int)
                cv2.drawContours(label_img, [point], -1, 255, thickness=-1)

        label_img[label_img > 0] = 255
        mask = label_img

        if is_aug_mask:
            mask_dilate = copy.deepcopy(mask)
            kernel = np.ones((19, 19), np.uint8)
            mask_dilate = cv2.dilate(mask_dilate, kernel)

            left_image, left_mask, left_dilate = copy.deepcopy(image), copy.deepcopy(mask), copy.deepcopy(mask_dilate)
            right_image, right_mask, right_dilate = copy.deepcopy(image), copy.deepcopy(mask), copy.deepcopy(mask_dilate)

            dist = int(70 * image_w / 512.0)

            if np.random.rand() < 0.5:     # left up move
                left_dilate[:-dist, :-dist] = left_dilate[dist:, dist:]
                left_image[:-dist, :-dist] = left_image[dist:, dist:]
                left_mask[:-dist, :-dist] = left_mask[dist:, dist:]
            else:   # left down move
                left_dilate[dist:, :-dist] = mask_dilate[:-dist, dist:]
                left_image[dist:, :-dist] = left_image[:-dist, dist:]
                left_mask[dist:, :-dist] = left_mask[:-dist, dist:]
            if np.random.rand() < 0.5:   # right up move
                right_dilate[:-dist, dist:] = right_dilate[dist:, :-dist]
                right_image[:-dist, dist:] = right_image[dist:, :-dist]
                right_mask[:-dist, dist:] = right_mask[dist:, :-dist]
            else:   # right down move
                right_dilate[dist:, dist:] = right_dilate[:-dist, :-dist]
                right_image[dist:, dist:] = right_image[:-dist, :-dist]
                right_mask[dist:, dist:] = right_mask[:-dist, :-dist]

            index = left_dilate > 0
            image[index, :] = left_image[index, :]
            mask[index] = left_mask[index]

            index = right_dilate > 0
            image[index, :] = right_image[index, :]
            mask[index] = right_mask[index]

            # print(mask.sum()/255/image_h/image_w)
            # cv2.imshow('new_image', image)
            # cv2.imshow('mask', mask)
            # cv2.waitKey()
    return image, mask, is_polygon


def create_dataset(file, save_path, train_val, times, rotate=False, crop=False, is_show=False, aug_mask=False):
    temp_times = times
    resize_w, resize_h = 512, 512
    path, name = os.path.split(file)

    if aug_mask:
        name = name.replace('.jpg', '_aug_mask.jpg')

    image, label_img, is_polygon = get_aug_mask(file, aug_mask)
    if is_polygon:
        train_val = train_val.replace('val/', 'train/')
        temp_times = 3*times

    image_train_val_dir = save_path + 'images/' + train_val
    ll_seg_annotations_dir = save_path + 'll_seg_annotations/' + train_val

    if not os.path.exists(image_train_val_dir):
        os.makedirs(image_train_val_dir)
    if not os.path.exists(ll_seg_annotations_dir):
        os.makedirs(ll_seg_annotations_dir)

    image = cv2.resize(image, (resize_w, resize_h))
    label_img = cv2.resize(label_img, (resize_w, resize_h), cv2.INTER_NEAREST)

    cv2.imwrite(ll_seg_annotations_dir + name.replace('.jpg', '.png'), label_img) #cv2.imwrite(save_path + 'll_seg_annotations/' + train_val+name.replace('.jpg', '.png'), label_img)
    cv2.imwrite(image_train_val_dir + name, image) # cv2.imwrite(save_path + 'images/' + train_val+name, image)

    if is_show:
        save_path_show = save_path + 'is_show/'
        if not os.path.exists(save_path_show):
            os.makedirs(save_path_show)
        show_label(image, label_img, save_path_show, name.replace('.json', '.jpg'))

    if rotate:
        agument_rotate(temp_times, copy.deepcopy(image), copy.deepcopy(label_img), image_train_val_dir, ll_seg_annotations_dir, name, resize_h, resize_w)

    if crop:
        agument_crop(temp_times, copy.deepcopy(image), copy.deepcopy(label_img), image_train_val_dir, ll_seg_annotations_dir, name, resize_h, resize_w)


def rgb2ind(im, color_path):
    # im = cv2.imread(gray_path, 0)
    im[im > 0] = 1
    gt = Image.fromarray(im)
    gt.putpalette(bin_colormap)
    gt.save(color_path)


def show_label(origin_image, label_image, save_path, name):
    color_area = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)
    color_area[label_image[:] == 255] = [0, 255, 0]
    color_seg = color_area
    # # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    color_image = copy.deepcopy(origin_image)
    color_image[color_mask != 0] = color_image[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    cv2.imwrite(save_path + name,  color_image)


def agument_rotate(times, image, label_img, image_train_val_dir, ll_seg_annotations_dir, name, resize_h, resize_w):
    for j in range(times):
        theter = random.randint(30, 150)
        rotate_img = rotateImageTheter(copy.deepcopy(image), theter)
        H, W, _ = rotate_img.shape
        crop_w = int((W - resize_w) * 0.5)
        crop_h = int((H - resize_h) * 0.5)
        rotate_img = rotate_img[crop_w:crop_w + resize_w, crop_h:crop_h + resize_h, :]
        cv2.imwrite(image_train_val_dir + name.replace('.jpg', '_' + str(j) + '.jpg'), rotate_img)

        rotate_label_img = rotateImageTheter(copy.deepcopy(label_img), theter)
        rotate_label_img = rotate_label_img[crop_w:crop_w + resize_w, crop_h:crop_h + resize_h]
        cv2.imwrite(ll_seg_annotations_dir + name.replace('.jpg', '_' + str(j) + '.png'), rotate_label_img)


def agument_crop(times, image, label_img, image_train_val_dir, ll_seg_annotations_dir, name, resize_h, resize_w):
    for j in range(times):
        theter = random.randint(30, 150)
        crop_num = random.randint(20, 100)
        times_num = j
        img_h, img_w, _ = image.shape
        label_h, label_w = label_img.shape
        img_crop = image
        label_crop = label_img
        if times_num == 1:
            img_crop = image[crop_num:img_w, :, :]
            label_crop = label_img[crop_num:label_w, :]
        if times_num == 2:
            img_crop = image[0:img_w - crop_num, :, :]
            label_crop = label_img[0:label_w - crop_num, :]
        if times_num == 3:
            img_crop = image[:, crop_num:img_h, :]
            label_crop = label_img[:, crop_num:label_h]
        if times_num == 4:
            img_crop = image[:, 0:img_h - crop_num, :]
            label_crop = label_img[:, 0:label_h - crop_num]

        img_crop_512 = cv2.resize(img_crop, (512, 512))
        label_crop_512 = cv2.resize(label_crop, (512, 512), cv2.INTER_NEAREST)

        rotate_img = rotateImageTheter(img_crop_512, theter)
        H, W, _ = rotate_img.shape
        crop_w = int((W - resize_w) * 0.5)
        crop_h = int((H - resize_h) * 0.5)
        rotate_img = rotate_img[crop_w:crop_w + resize_w, crop_h:crop_h + resize_h, :]
        cv2.imwrite(image_train_val_dir + name.replace('.jpg', '_crop_' + str(j) + '.jpg'), rotate_img)

        rotate_label_img = rotateImageTheter(label_crop_512, theter)
        rotate_label_img = rotate_label_img[crop_w:crop_w + resize_w, crop_h:crop_h + resize_h]
        cv2.imwrite(ll_seg_annotations_dir + name.replace('.jpg', '_crop_' + str(j) + '.png'), rotate_label_img)


def find_error_label():
    save_path = '/media/z590/D/DataSet/BYD_parking_slot/finish_label/lane_parking_line/error/'
    data_path = '/media/z590/D/DataSet/BYD_parking_slot/finish_label/lane_parking_line/8_month_line_label/'  #
    exts = ['.json']
    file_lists = get_files(data_path, exts)
    for i, file in enumerate(file_lists):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_lists)))
        sys.stdout.flush()
        path, name = os.path.split(file)
        img = cv2.imread(file.replace('.json', '.jpg'))
        H, W, _ = img.shape
        binary00 = np.zeros((H, W), dtype=np.uint8)

        labels, points, shape_types = read_json(file)
        for j, shape_type in enumerate(shape_types):
            point = []
            binary = np.zeros((H, W), dtype=np.uint8)
            if shape_type == 'polygon':
                xxx = points[j]
                for p in xxx:
                    point.append([int(p[0]), int(p[1])])
                cv2.drawContours(binary, [np.array(point)], 0, 120, cv2.FILLED)
                binary00 += binary

        binary00[binary00 < 200] = 0
        binary00[binary00 > 200] = 1
        pixs = np.sum(binary00)
        if pixs > 1000:
            shutil.copy(file, save_path + name)
            shutil.copy(file.replace('.json', '.jpg'), save_path + name.replace('.json', '.jpg'))


def transform2VOC():
    data_path = '/media/z590/D/DataSet/BYD_parking_slot/train_dataSet/lane_parking_line/1027_test/'  #
    root = data_path+'VOC2012/'
    JPEGImages = root + 'JPEGImages/'
    if not os.path.exists(JPEGImages):
        os.makedirs(JPEGImages)
    SegmentationClass = root + 'SegmentationClass/'
    if not os.path.exists(SegmentationClass):
        os.makedirs(SegmentationClass)
    txt_path = root + 'ImageSets/Segmentation/'
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    fid_train = open(txt_path+'/train.txt', 'w')
    fid_val = open(txt_path+'/val.txt', 'w')

    exts = ['.png']
    file_lists = get_files(data_path, exts)
    for i, file in enumerate(file_lists):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_lists)))
        sys.stdout.flush()
        path, name = os.path.split(file)
        # label to VOC label
        im = cv2.imread(file, 0)
        im[im > 0] = 1
        gt = Image.fromarray(im)
        gt.putpalette(bin_colormap)
        gt.save(SegmentationClass+name)
        # copy file
        shutil.copy(file.replace('.png', '.jpg').replace('ll_seg_annotations', 'images'), JPEGImages+name.replace('.png', '.jpg'))
        if 'val' in path:
            fid_val.write(name.replace('.png', '') + "\n")
        else:
            fid_train.write(name.replace('.png', '') + "\n")
    fid_train.close()
    fid_val.close()


def save_voc(image, label, name, trainval, SegmentationClass, JPEGImages, fid_train, fid_val):
    label[label > 0] = 1
    gt = Image.fromarray(label)
    gt.putpalette(bin_colormap)
    gt.save(SegmentationClass + name.replace('.jpg', '.png'))

    cv2.imwrite(JPEGImages + name, image)

    if 'train' in trainval:
        fid_train.write(name.replace('.jpg', '') + "\n")
    else:
        fid_val.write(name.replace('.jpg', '') + "\n")


if __name__ == '__main__':
    # find_error_label()
    # transform2VOC()
    # file = '/media/z590/D/DataSet/BYD_parking_slot/finish_label/lane_parking_line/1014/same/2022.4.14_CZ-N6_95.jpg'
    # get_aug_mask(file, True)

    # main(aug_mask=True)
    main(aug_mask=False)



