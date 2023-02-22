import numpy as np
import cv2
import random
from PIL import Image
import copy

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if np.random.rand() > 0.7:
        image = image[..., ::-1]
    else:
        if np.random.rand() > 0.7:
            image = augment_sunset_effect(image)
    return image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    w, h = size
    new_image = cv2.resize(image, (w, h))
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#
#---------------------------------------------------#
def preprocess_input(inputs):
    MEANS = (104, 117, 123)
    return inputs - MEANS

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def augment_salt_noise(subImage):
    if np.random.rand() < 0.7:
        return subImage
    height, width, _ = np.shape(subImage)
    rate = 1.0 * np.random.randint(0, 3) / 100.0
    for jj in range(int(width*height*rate)):
        row = np.random.randint(0, width-1)
        col = np.random.randint(0, height-1)
        value = np.random.randint(0, 255)
        subImage[col][row][:] = value
    return subImage


def augment_blur_noise(subImage):
    if np.random.rand() < 0.7:
        return subImage
    rand =np.random.randint(1, 3)
    subImage =cv2.blur(subImage, (rand ,rand))
    return subImage


def augment_contrast_brightness(subImage):
    if np.random.rand() < 0.7:
        return subImage
    c = 1  # C 是对比度
    val = np.random.randint(50, 200)
    gray_img = cv2.cvtColor(subImage, cv2.COLOR_BGR2GRAY)
    gray_val = np.mean(gray_img[:])
    b = int(val - gray_val)     # b 是亮度
    h, w, ch = subImage.shape
    blank = np.zeros([h, w, ch], subImage.dtype)
    dst = cv2.addWeighted(subImage, c, blank, 1 - c, b)  # 改变像素的API

    return dst


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    if np.random.rand() < 0.7:
        return img

    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
    return img


# 图像的日落效果
# 将蓝色值和绿色值设为原来的70 %, 红色值不变
# epoch = 0
def augment_sunset_effect(img):
    if np.random.rand() < 0.7:
        return img

    val = random.randint(6, 9)
    val = 0.1 * val
    img[:, :, 0] = img[:, :, 0] * val
    img[:, :, 1] = img[:, :, 1] * val

    # Height, Width, _ = img.shape
    # Height, Width = int(0.5*Height), int(0.5*Width)
    # zeros = np.zeros([724, 724], dtype='uint8')
    # random_H = random.randint(0, 724)
    # zeros[0:random_H, :] = 1
    # theter = random.randint(0, 180)
    # zeros = Image.fromarray(zeros)  # OpenCV转换成PIL.Image格式
    # im_rotate = zeros.rotate(theter, expand=False)
    # im_rotate = np.array(im_rotate)
    # H, W = im_rotate.shape
    # center_x, center_y = int(0.5*W), int(0.5*H)
    # crop_img = im_rotate[center_y-Height: center_y+Height, center_x-Width: center_x+Width]
    # img0 = img * cv2.merge([crop_img, crop_img, crop_img])
    # crop_img = 1 - crop_img
    # img1 = img1 * cv2.merge([crop_img, crop_img, crop_img])
    # img2 = img0 + img1

    return img


def merge_color_difference(img, img1):
    if np.random.rand() < 0.7:
        return img

    Height, Width, _ = img.shape
    Height, Width = int(0.5*Height), int(0.5*Width)
    zeros = np.zeros([724, 724], dtype='uint8')
    random_H = random.randint(0, 724)
    zeros[0:random_H, :] = 1
    theter = random.randint(0, 180)
    zeros = Image.fromarray(zeros)  # OpenCV转换成PIL.Image格式
    im_rotate = zeros.rotate(theter, expand=False)
    im_rotate = np.array(im_rotate)
    H, W = im_rotate.shape
    center_x, center_y = int(0.5 * W), int(0.5 * H)
    crop_img = im_rotate[center_y - Height: center_y + Height, center_x - Width: center_x + Width]
    img0 = img * cv2.merge([crop_img, crop_img, crop_img])
    crop_img = 1 - crop_img
    img1 = img1 * cv2.merge([crop_img, crop_img, crop_img])
    img2 = img0 + img1
    # cv2.imshow('img', img)
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # cv2.waitKey()

    return img2


# 在图片中心区域附件，把两侧往中间移动
def two_side_move_to_middle(image, boxes):
    if np.random.rand() < 0.7:
        return image, boxes

    flag = False
    for box in boxes:
        if (210 < box[0] and box[0] < 300) or (210 < box[2] and box[2] < 300):
            flag = True
    if not flag:
        move_left_dist = np.random.randint(5, 50)
        move_right_dist = np.random.randint(5, 50)
        image[:, move_left_dist:210, :] = image[:, 0:210-move_left_dist, :]
        image[:, :move_left_dist, :] = (0, 0, 0)
        image[:, 300:-move_right_dist, :] = image[:, 300+move_right_dist:, :]
        image[:, -move_right_dist:, :] = (0, 0, 0)
        for i, box in enumerate(boxes):
            if boxes[i][0] < 210:
                boxes[i][0] = boxes[i][0] + move_left_dist
                boxes[i][2] = boxes[i][2] + move_left_dist
            else:
                boxes[i][0] = boxes[i][0] - move_right_dist
                boxes[i][2] = boxes[i][2] - move_right_dist
            # cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 255), 3)
        # cv2.imshow('image_cropped', image)
        # cv2.waitKey()

    return image, boxes


#------------------------------------------------------------------------------
#   Rotate angle
#------------------------------------------------------------------------------
def rotate(ps, M):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(M, pts)
    target_point = [[int(target_point[0][x]),int(target_point[1][x])] for x in range(len(target_point[0]))]
    return target_point[0]


def rotate_angle(image, boxes, size=512, angle_max=80):
    if np.random.rand() < 0.7:
        return image, boxes

    angle = np.random.randint(-angle_max, angle_max)
    # Get parameters for affine transform
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform transform
    new_image = cv2.warpAffine(image, M, (nW, nH))
    (new_h, new_w) = new_image.shape[:2]
    # crop
    start_x = int(0.5 * (new_w - size))
    start_y = int(0.5 * (new_h - size))
    image_cropped = new_image[start_y:start_y + size, start_x:start_x + size, :]
    new_boxes = []
    for i, box in enumerate(boxes):
        ps0 = rotate([box[0], box[1]], M)
        ps1 = rotate([box[2], box[1]], M)
        ps2 = rotate([box[2], box[3]], M)
        ps3 = rotate([box[0], box[3]], M)
        min_x = min(ps0[0], ps1[0], ps2[0], ps3[0])
        min_y = min(ps0[1], ps1[1], ps2[1], ps3[1])
        max_x = max(ps0[0], ps1[0], ps2[0], ps3[0])
        max_y = max(ps0[1], ps1[1], ps2[1], ps3[1])
        if (start_x < min_x and min_x < start_x + size) and \
                (start_y < min_y and min_y < start_y + size) and \
                (start_x < max_x and max_x < start_x + size) and \
                (start_y < max_y and max_y < start_y + size):
            new_boxes.append([min_x-start_x, min_y-start_y, max_x-start_x, max_y-start_y, box[4]])
            # cv2.rectangle(image_cropped, (min_x-start_x, min_y-start_y), (max_x-start_x, max_y-start_y), (0, 0, 255), 3)
    # cv2.imshow('image_cropped', image_cropped)
    # cv2.waitKey()
    # print(new_boxes)
    return image_cropped, np.array(new_boxes)


def augments(img, boxes):
    flag = False
    if np.random.rand() > 0.7:
        image = img[..., ::-1]
        # cv2.imshow('image-1', image)
    else:
        if np.random.rand() > 0.7:
            if np.random.rand() > 0.5:
                image = augment_sunset_effect(copy.deepcopy(img))
            else:
                image = img
                flag = True
        else:
            image = img
        # cv2.imshow('image', image)
    # cv2.imshow('img', img)
    # cv2.waitKey()

    image = augment_salt_noise(copy.deepcopy(image))
    image = augment_blur_noise(copy.deepcopy(image))
    image = augment_contrast_brightness(copy.deepcopy(image))
    image = augment_hsv(copy.deepcopy(image))
    if flag:
        image = augment_sunset_effect(copy.deepcopy(image))

    image = merge_color_difference(image, img)

    image, boxes = two_side_move_to_middle(copy.deepcopy(image), boxes)

    image, boxes = rotate_angle(image, boxes)

    return image, boxes

