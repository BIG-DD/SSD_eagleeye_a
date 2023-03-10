import copy
import time

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
from multiprocessing import Pool
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(description='Generated data')
    parser.add_argument('--rotate',
                        default=False,
                        type=bool,
                        help='Image rotation enhancement')
    parser.add_argument('--times',
                        type=int,
                        default=3,
                        help='Data increment times')
    parser.add_argument('--vertical_transform_incline',
                        type=bool,
                        default=False,
                        help='Image vertical transform incline')
    parser.add_argument('--image_json_dir',
                        type=str,
                        default='D:/data/8_23_normal/jpg_json_0_20/',
                        help='Image and json dir for make mask')
    parser.add_argument('--origin_json_dir',
                        type=str,
                        default='D:/data/6_22_data/all_data_origin/',
                        help='get vertical label')
    parser.add_argument('--save_dir',
                        type=str,
                        default='D:/data/8_23_normal/dataset_0_20_s/',
                        help='data directory')


    args = parser.parse_args()

    return args

def main(arg, save_path, data_path, times, origin_json, file, i):
    # random.shuffle(file_lists)
    # for i, file in enumerate(file_lists):
    # sys.stdout.write('\r>> Converting images %d/%d' % (i + 1, len(file_lists)))
    # sys.stdout.flush()
    path, name = os.path.split(file)
    # if '1112ps4000015' not in name:
    #     continue

    # if i % 8 == 0:
    #     train_val = 'val/'
    # else:
    #     train_val = 'train/'
    train_val = 'train/'

    # add_watershed_algorithm(file, file.replace('.jpg', '.json'), save_path, name, train_val, resize_w=512, resize_h=512, is_show=False)
    json_dir = origin_json + name.replace('.jpg', '.json')
    json_dir1 = data_path + name.replace('.jpg', '.json')
    if os.path.exists(json_dir1):
        _, _, shape_type, _, _ = read_json_wh(json_dir1)
        if 'polygon' in shape_type:
            times = 6
        else:
            times = 3
    if os.path.exists(json_dir):
        if arg.vertical_transform_incline:
            vertical_transform_incline(json_dir, file, save_path, train_val, resize_w=512, resize_h=512, angles=[-60,-50, -40,-30, -20, -10, 10, 20, 30, 40, 50, 60])
    if arg.rotate:
        create_dataset(file, save_path, train_val, times)
    if not arg.rotate:
        create_dataset_not_rotate(file, save_path, train_val)
        # create_dataset_not_rotate_img(file, save_path, train_val) # 在原图上把把标签位置涂黑

def vertical_transform_incline(json_file, mask_file, save_path, train_val_tr, resize_w=512, resize_h=512, angles=[-60,-50, -40,-30, -20, -10, 10, 20, 30, 40, 50, 60]):
    lane_lines = []
    vertical_points = []
    is_vertical = False
    is_horizon = False
    path, name = os.path.split(json_file)#原始文件
    img_path = str(json_file).replace('.json', '.jpg')
    image = cv2.imread(img_path)
    # height, width, _ = image.shape
    # width = 554
    # height = 720

    labels, points, shape_types, width, height = read_json_wh(json_file)
    gt_mask = get_mask(mask_file)
    gt_mask = cv2.resize(gt_mask, (width, height))
    image = cv2.resize(image, (width, height))

    for j, point in enumerate(points):
        if len(point) > 1 and labels[j] == 'lane':
            for k in range(len(point) - 1):
                lane_lines.append([[int(point[k][0]), int(point[k][1])], [int(point[k + 1][0]), int(point[k + 1][1])]])

        elif labels[j] == 'vertical' and len(point) == 2:
            vertical_points.append([int(point[0][0]), int(point[0][1])])
            vertical_points.append([int(point[1][0]), int(point[1][1])])
            if abs(point[0][0] - point[1][0]) < 5:
                is_vertical = True
            elif abs(point[0][1] - point[1][1]) < 5:
                is_horizon = True
    if is_horizon or is_vertical and not (is_horizon and is_vertical):
        image1 = copy.deepcopy(image)
        # 计算车位线角点，确定透视变换点
        if is_vertical:
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
            is_save = True
            if len(left_mid_x) > 0:
                std_x = np.std(left_mid_x)
                print(std_x)
                if std_x < 10:
                    is_save = True
                    left_mid_x = int(np.mean(left_mid_x))
                    src = np.array([[left_mid_x, left_min_y], [left_mid_x, left_max_y], [0, left_max_y], [0, left_min_y]],
                                   np.float32)
                    theter = choice(angles)

                    padding = (left_mid_x) * np.tan(theter * np.pi / 180)
                    dst = np.array([[left_mid_x, left_min_y], [left_mid_x, left_max_y], [0, left_max_y + padding],
                                    [0, left_min_y + padding]],
                                   np.float32)
                    crop_img = image[:, 0:left_mid_x, :]
                    result = transform(copy.deepcopy(crop_img), src, dst, height, left_mid_x)
                    image1[:, 0:left_mid_x, :] = result

                    crop_img = gt_mask[:, 0:left_mid_x]
                    result = transform(copy.deepcopy(crop_img), src, dst, height, left_mid_x)
                    result[result>0]=255
                    gt_mask[:, 0:left_mid_x] = result

            if len(right_mid_x) > 0:
                std_x = np.std(right_mid_x)
                print(std_x)
                if std_x < 10:
                    is_save = True
                    right_mid_x = int(np.mean(right_mid_x))
                    src = np.array([[0, right_min_y], [0, right_max_y], [width - right_mid_x, right_max_y],
                                    [width - right_mid_x, right_min_y]], np.float32)
                    theter = choice(angles)
                    padding = (width - right_mid_x) * np.tan(theter * np.pi / 180)

                    dst = np.array([[0, right_min_y], [0, right_max_y], [width - right_mid_x, right_max_y + padding],
                                    [width - right_mid_x, right_min_y + padding]], np.float32)
                    crop_img = image[:, right_mid_x:, :]
                    result = transform(copy.deepcopy(crop_img), src, dst, height, width - right_mid_x)
                    image1[:, right_mid_x:, :] = result

                    crop_img = gt_mask[:, right_mid_x:]
                    result = transform(copy.deepcopy(crop_img), src, dst, height, width - right_mid_x)
                    result[result > 0] = 255
                    gt_mask[:, right_mid_x:] = result
            image1 = cv2.resize(image1, (resize_w, resize_h))
            gt_mask = cv2.resize(gt_mask, (resize_w, resize_h))

            image_train_val_dir = save_path + 'images/' + train_val_tr
            ll_seg_annotations_dir = save_path + 'll_seg_annotations/' + train_val_tr

            if not os.path.exists(image_train_val_dir):
                os.makedirs(image_train_val_dir)
            if not os.path.exists(ll_seg_annotations_dir):
                os.makedirs(ll_seg_annotations_dir)

            cv2.imwrite(image_train_val_dir + name.replace('.json', '_inc_0.jpg'), image1)
            cv2.imwrite(ll_seg_annotations_dir +name.replace('.json', '_inc_0.png'), gt_mask)

            for j in range(1, 4):
                theter = random.randint(30, 150)
                rotate_img = rotateImageTheter(copy.deepcopy(image1), theter)
                H, W, _ = rotate_img.shape
                # rotate_img = cv2.resize(rotate_img, (image_w, image_h))
                crop_w = int((W-resize_w)*0.5)
                crop_h = int((H-resize_h)*0.5)
                rotate_img = rotate_img[crop_w:crop_w+resize_w, crop_h:crop_h+resize_h, :]
                cv2.imwrite(image_train_val_dir + name.replace('.json', '_inc_' + str(j) + '.jpg'), rotate_img)

                rotate_label_img = rotateImageTheter(copy.deepcopy(gt_mask), theter)
                # rotate_label_img = cv2.resize(rotate_label_img, (image_w, image_h))
                rotate_label_img = rotate_label_img[crop_w:crop_w + resize_w, crop_h:crop_h + resize_h]
                cv2.imwrite(ll_seg_annotations_dir + name.replace('.json', '_inc_' + str(j) + '.png'), rotate_label_img)

def read_json_wh(file):
    points = []
    labels = []
    shape_types = []
    with open(file, 'r', encoding='utf-8') as f:
        setting = json.load(f)
    coordinate = setting['shapes']
    height = setting['imageHeight']
    width = setting['imageWidth']

    for j in range(len(coordinate)):
        label = coordinate[j]['label']
        point = coordinate[j]['points']
        shape_type = coordinate[j]['shape_type']
        if shape_type == 'rectangle':
            continue
        labels.append(label)
        points.append(point)
        shape_types.append(shape_type)

    return labels, points, shape_types, width, height

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
    resize_w, resize_h = 512, 512
    image = cv2.imread(file)
    image_h, image_w, _ = image.shape
    h_w_max = max(image_h, image_w)
    thickness_value = int(h_w_max*(5/512))
    label_img = np.zeros((image_h, image_w), dtype=np.uint8)

    if not os.path.exists(file.replace('.jpg', '.json')):
        mask = label_img
    else:
        labels, points, shape_types = read_json(file.replace('.jpg', '.json'))
        for j, point in enumerate(points):
            if len(point) > 1 and shape_types[j] == 'line':
                for k in range(len(point) - 1):
                    cv2.line(label_img, (int(point[k][0]), int(point[k][1])),
                             (int(point[k + 1][0]), int(point[k + 1][1])),
                             255, thickness=thickness_value) #改

            if shape_types[j] == 'polygon':
                point = np.array(point, dtype=int)
                # cv2.drawContours(label_img, [point], -1, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.drawContours(label_img, [point], -1, 255, thickness=-1)
        label_img = cv2.resize(label_img, (resize_w, resize_h))
        _, binary = cv2.threshold(label_img, 50, 255, cv2.THRESH_BINARY)
        mask = binary
    return mask

def create_dataset(file, save_path, train_val, times):
    resize_w, resize_h = 512, 512
    path, name = os.path.split(file)
    image = cv2.imread(file)
    image_h, image_w, _ = image.shape
    image = cv2.resize(image, (resize_w, resize_h))
    label_img = np.zeros((image_h, image_w), dtype=np.uint8)
    image_train_val_dir = save_path + 'images/' + train_val
    ll_seg_annotations_dir = save_path + 'll_seg_annotations/' + train_val

    if not os.path.exists(image_train_val_dir):
        os.makedirs(image_train_val_dir)
    if not os.path.exists(ll_seg_annotations_dir):
        os.makedirs(ll_seg_annotations_dir)

    cv2.imwrite(image_train_val_dir + name, image) # cv2.imwrite(save_path + 'images/' + train_val+name, image)
    if not os.path.exists(file.replace('.jpg', '.json')):
        label_img = cv2.resize(label_img, (resize_w, resize_h))
        cv2.imwrite(ll_seg_annotations_dir +name.replace('.jpg', '.png'), label_img) #cv2.imwrite(save_path + 'll_seg_annotations/' + train_val+name.replace('.jpg', '.png'), label_img)
    else:
        labels, points, shape_types = read_json(file.replace('.jpg', '.json'))
        for j, point in enumerate(points):
            if len(point) > 1 and shape_types[j] == 'line':
                for k in range(len(point) - 1):
                    cv2.line(label_img, (int(point[k][0]), int(point[k][1])),
                             (int(point[k + 1][0]), int(point[k + 1][1])),
                             255, thickness=5)

            if shape_types[j] == 'polygon':
                point = np.array(point, dtype=int)
                cv2.drawContours(label_img, [point], -1, 255, thickness=-1)
        label_img = cv2.resize(label_img, (resize_w, resize_h))
        _, label_img = cv2.threshold(label_img, 50, 255, cv2.THRESH_BINARY)
        cv2.imwrite(ll_seg_annotations_dir + name.replace('.jpg', '.png'), label_img) #cv2.imwrite(save_path + 'll_seg_annotations/' + train_val+name.replace('.jpg', '.png'), label_img)

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

    for j in range(1, 5):
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
            img_crop = image[0:img_w-crop_num, :, :]
            label_crop = label_img[0:label_w-crop_num, :]
        if times_num == 3:
            img_crop = image[:, crop_num:img_h, :]
            label_crop = label_img[:, crop_num:label_h]
        if times_num == 4:
            img_crop = image[:, 0:img_h-crop_num, :]
            label_crop = label_img[:, 0:label_h-crop_num]


        img_crop_512 = cv2.resize(img_crop, (512, 512))
        label_crop_512 = cv2.resize(label_crop, (512, 512))

        rotate_img = rotateImageTheter(img_crop_512, theter)
        H, W, _ = rotate_img.shape
        crop_w = int((W - resize_w) * 0.5)
        crop_h = int((H - resize_h) * 0.5)
        rotate_img = rotate_img[crop_w:crop_w + resize_w, crop_h:crop_h + resize_h, :]
        cv2.imwrite(image_train_val_dir + name.replace('.jpg', '_crop_' + str(j) + '.jpg'), rotate_img)

        rotate_label_img = rotateImageTheter(label_crop_512, theter)
        rotate_label_img = rotate_label_img[crop_w:crop_w + resize_w, crop_h:crop_h + resize_h]
        cv2.imwrite(ll_seg_annotations_dir + name.replace('.jpg', '_crop_' + str(j) + '.png'), rotate_label_img)

def create_dataset_not_rotate(file, save_path, train_val):
    resize_w, resize_h = 512, 512
    path, name = os.path.split(file)
    image = cv2.imread(file)
    image_h, image_w, _ = image.shape
    image = cv2.resize(image, (resize_w, resize_h))
    label_img = np.zeros((image_h, image_w), dtype=np.uint8)
    image_train_val_dir = save_path + 'images/' + train_val
    ll_seg_annotations_dir = save_path + 'll_seg_annotations/' + train_val

    if not os.path.exists(image_train_val_dir):
        os.makedirs(image_train_val_dir)
    if not os.path.exists(ll_seg_annotations_dir):
        os.makedirs(ll_seg_annotations_dir)

    cv2.imwrite(image_train_val_dir + name, image) # cv2.imwrite(save_path + 'images/' + train_val+name, image)
    if not os.path.exists(file.replace('.jpg', '.json')):
        label_img = cv2.resize(label_img, (resize_w, resize_h))
        # time.sleep(0.01)
        cv2.imwrite(ll_seg_annotations_dir +name.replace('.jpg', '.png'), label_img) #cv2.imwrite(save_path + 'll_seg_annotations/' + train_val+name.replace('.jpg', '.png'), label_img)
    else:
        labels, points, shape_types = read_json(file.replace('.jpg', '.json'))
        for j, point in enumerate(points):
            if len(point) > 1 and shape_types[j] == 'line':
                for k in range(len(point) - 1):
                    cv2.line(label_img, (int(point[k][0]), int(point[k][1])),
                             (int(point[k + 1][0]), int(point[k + 1][1])),
                             255, thickness=5)

            if shape_types[j] == 'polygon':
                point = np.array(point, dtype=int)
                cv2.drawContours(label_img, [point], -1, 255, thickness=-1)

        label_img = cv2.resize(label_img, (resize_w, resize_h))
        _, label_img = cv2.threshold(label_img, 50, 255, cv2.THRESH_BINARY)
        # time.sleep(0.01)
        cv2.imwrite(ll_seg_annotations_dir + name.replace('.jpg', '.png'), label_img) #cv2.imwrite(save_path + 'll_seg_annotations/' + train_val+name.replace('.jpg', '.png'), label_img)

def create_dataset_not_rotate_img(file, save_path, train_val):
    # 把车位线用黑色覆盖
    resize_w, resize_h = 512, 512
    path, name = os.path.split(file)
    image = cv2.imread(file)
    image_h, image_w, _ = image.shape
    # image = cv2.resize(image, (resize_w, resize_h))
    label_img = np.zeros((image_h, image_w), dtype=np.uint8)
    image_train_val_dir = save_path + 'images/' + train_val
    ll_seg_annotations_dir = save_path + 'll_seg_annotations/' + train_val

    if not os.path.exists(image_train_val_dir):
        os.makedirs(image_train_val_dir)
    if not os.path.exists(ll_seg_annotations_dir):
        os.makedirs(ll_seg_annotations_dir)

    cv2.imwrite(image_train_val_dir + name, image) # cv2.imwrite(save_path + 'images/' + train_val+name, image)
    if not os.path.exists(file.replace('.jpg', '.json')):
        label_img = cv2.resize(label_img, (resize_w, resize_h))
        cv2.imwrite(ll_seg_annotations_dir +name, image) #cv2.imwrite(save_path + 'll_seg_annotations/' + train_val+name.replace('.jpg', '.png'), label_img)
    else:
        labels, points, shape_types = read_json(file.replace('.jpg', '.json'))
        for j, point in enumerate(points):
            if len(point) > 1 and shape_types[j] == 'line':
                for k in range(len(point) - 1):
                    cv2.line(image, (int(point[k][0]), int(point[k][1])),
                             (int(point[k + 1][0]), int(point[k + 1][1])),
                             (0, 0, 0), thickness=5)

            if shape_types[j] == 'polygon':
                point = np.array(point, dtype=int)
                cv2.drawContours(image, [point], -1, (0, 0, 0), thickness=-1)

        label_img = cv2.resize(label_img, (resize_w, resize_h))
        _, label_img = cv2.threshold(label_img, 50, 255, cv2.THRESH_BINARY)
        cv2.imwrite(ll_seg_annotations_dir + name, image) #cv2.imwrite(save_path + 'll_seg_annotations/' + train_val+name.replace('.jpg', '.png'), label_img)

def multi_process_list(nums, file_list):
    multi_list = []
    every_num = len(file_list)//nums
    for i in range(nums-1):
        multi_list.append(file_list[i*every_num:(i+1)*every_num])
    multi_list.append(file_list[(nums-1)*every_num:])
    return multi_list

if __name__ == '__main__':
    start_time = time.time()
    arg = parse_args()
    save_path = arg.save_dir
    data_path = arg.image_json_dir
    times = arg.times
    origin_json = arg.origin_json_dir
    exts = ['jpg']
    file_lists = get_files(data_path, exts)

    # for i, file in enumerate(file_lists):
    #     print(i)
    #     main(arg, save_path, data_path, times, origin_json, file, i,)

    multiprocessing.freeze_support()
    p = Pool()
    for i, file in enumerate(file_lists):
        print(i)
        p.apply_async(main, args=(arg, save_path, data_path, times, origin_json, file, i, ))
    print('等待所有进程结束')
    p.close()
    p.join()
    print('主进程结束')

    end_time = time.time()
    print(end_time-start_time)