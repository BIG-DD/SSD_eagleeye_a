#------------------------------------------------------------------------------
#   Library
#------------------------------------------------------------------------------
import cv2
import numpy as np
import random
from PIL import Image
import copy

random_rate = 0.7


#------------------------------------------------------------------------------
#   Random crop
#------------------------------------------------------------------------------
def random_crop(image, label, crop_range, size=512):
    """
    cropped image is a square.
    image (ndarray) with shape [H,W,3]
    label (ndarray) with shape [H,W]
    crop_ratio (list) contains 2 bounds
    """
    ##### Exception #####
    if np.random.rand() < random_rate:
        H, W = label.shape
        start_x = int(0.5*(W - size))
        start_y = int(0.5*(H - size))

        image_cropped = image[start_y:start_y + size, start_x:start_x + size, :]
        label_cropped = label[start_y:start_y + size, start_x:start_x + size]

        return image_cropped, label_cropped

    if crop_range[0] == crop_range[1] and crop_range[0] == 1.0:
        return image, label

    # # Get random crop_ratio
    # crop_ratio = np.random.choice(np.linspace(crop_range[0], crop_range[1], num=10), size=())

    # Get random coordinates
    H, W = label.shape
    # size = H if H < W else W
    # size = int(size*crop_ratio)

    max_i, max_j = H-size, W-size
    i = np.random.choice(np.arange(0, max_i+1), size=())
    j = np.random.choice(np.arange(0, max_j+1), size=())

    # Crop
    image_cropped = image[i:i+size, j:j+size, :]
    label_cropped = label[i:i+size, j:j+size]
    return image_cropped, label_cropped


#------------------------------------------------------------------------------
#   Horizontal or vertical flip
#------------------------------------------------------------------------------
def flip_horizon(image, label):
    if np.random.rand() < random_rate:
        return image, label

    if np.random.rand() < 0.5:
        image = np.flip(image, axis=0)	 # up down
        label = np.flip(label, axis=0)
    else:
        image = np.flip(image, axis=1)	 # left right
        label = np.flip(label, axis=1)

    return image, label


#------------------------------------------------------------------------------
#   Rotate 90
#------------------------------------------------------------------------------
def rotate_90(image, label):
    if np.random.rand() < random_rate:
        return image, label

    k = np.random.choice([-1, 0, 1])
    if k:
        image = np.rot90(image, k=k, axes=(0,1))
        label = np.rot90(label, k=k, axes=(0,1))
    return image, label


#------------------------------------------------------------------------------
#   Rotate angle
#------------------------------------------------------------------------------
def rotate_angle(image, label, angle_max):
    if np.random.rand() < random_rate:
        return image, label
    if angle_max:
        # Random angle in range [-angle_max, angle_max]
        # angle = np.random.choice(np.linspace(-angle_max, angle_max, num=21), size=())
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
        image = cv2.warpAffine(image, M, (nW, nH))
        label = cv2.warpAffine(label, M, (nW, nH))
    return image, label


#------------------------------------------------------------------------------
#  Gaussian noise
#------------------------------------------------------------------------------
def random_noise(image, std):
    if np.random.rand() < random_rate:
        return image
    if std:
        noise = np.random.normal(0, std, size=image.shape)
        image = image + noise
        image[image < 0] = 0
        image[image > 255] = 255
        image = image.astype(np.uint8)
    return image


#------------------------------------------------------------------------------
#  Resize image
#------------------------------------------------------------------------------
def resize_image(image, expected_size, pad_value, ret_params=False, mode=cv2.INTER_LINEAR):
    """
    image (ndarray) with either shape of [H,W,3] for RGB or [H,W] for grayscale.
    Padding is added so that the content of image is in the center.
    """
    h, w = image.shape[:2]
    if w>h:
        w_new = int(expected_size)
        h_new = int(h * w_new / w)
        image = cv2.resize(image, (w_new, h_new), interpolation=mode)

        pad_up = (w_new - h_new) // 2
        pad_down = w_new - h_new - pad_up
        if len(image.shape)==3:
            pad_width = ((pad_up, pad_down), (0,0), (0,0))
            constant_values=((pad_value, pad_value), (0,0), (0,0))
        elif len(image.shape)==2:
            pad_width = ((pad_up, pad_down), (0,0))
            constant_values=((pad_value, pad_value), (0,0))

        image = np.pad(
            image,
            pad_width=pad_width,
            mode="constant",
            constant_values=constant_values,
        )
        if ret_params:
            return image, pad_up, 0, h_new, w_new
        else:
            return image

    elif w<h:
        h_new = int(expected_size)
        w_new = int(w * h_new / h)
        image = cv2.resize(image, (w_new, h_new), interpolation=mode)

        pad_left = (h_new - w_new) // 2
        pad_right = h_new - w_new - pad_left
        if len(image.shape)==3:
            pad_width = ((0,0), (pad_left, pad_right), (0,0))
            constant_values=((0,0), (pad_value, pad_value), (0,0))
        elif len(image.shape)==2:
            pad_width = ((0,0), (pad_left, pad_right))
            constant_values=((0,0), (pad_value, pad_value))

        image = np.pad(
            image,
            pad_width=pad_width,
            mode="constant",
            constant_values=constant_values,
        )
        if ret_params:
            return image, 0, pad_left, h_new, w_new
        else:
            return image

    else:
        image = cv2.resize(image, (expected_size, expected_size), interpolation=mode)
        if ret_params:
            return image, 0, 0, expected_size, expected_size
        else:
            return image


def augment_salt_noise(subImage):
    if np.random.rand() < random_rate:
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
    if np.random.rand() < random_rate:
        return subImage
    rand = np.random.randint(1, 3)
    dst = cv2.blur(subImage, (rand ,rand))
    return dst


def augment_contrast_brightness(subImage):
    if np.random.rand() < random_rate:
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
    # """change color hue, saturation, value"""
    if np.random.rand() < random_rate:
        return img
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
    return img


# 图像的日落效果
# 将蓝色值和绿色值设为原来的70 %, 红色值不变,BGR
def augment_sunset_effect(img):
    if np.random.rand() < random_rate:
        return img

    val = random.randint(6, 9)
    val = 0.1*val
    img1 = copy.deepcopy(img)
    img1[:, :, 0] = img1[:, :, 0]*val
    img1[:, :, 1] = img1[:, :, 1]*val

    Height, Width, _ = img.shape
    Height, Width = int(0.5 * Height), int(0.5 * Width)

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

    return img2


def augments(image, label, angle, crop_range):
    flag = False
    if np.random.rand() > random_rate:
        image = image[..., ::-1]
    else:
        if np.random.rand() > random_rate:
            if np.random.rand() > 0.5:
                image = augment_sunset_effect(image)
            else:
                flag = True

    image = augment_salt_noise(image)
    image = augment_blur_noise(image)
    image = augment_contrast_brightness(image)
    image = augment_hsv(image)
    if flag:
        image = augment_sunset_effect(image)

    image, label = two_side_move_to_middle(image, label)
    # image = transforms.random_noise(image, std=self.noise_std)
    image, label = flip_horizon(image, label)
    image, label = rotate_90(image, label)
    image, label = rotate_angle(image, label, angle)
    image, label = random_crop(image, label, crop_range)
    return image, label


# 在图片中心区域附件，把两侧往中间移动
def two_side_move_to_middle(image, label):
    if np.random.rand() < random_rate:
        return image, label

    # x[210:300],y[161:351], W=90,H=190
    if np.sum(label[:, 210:300]) == 0:
        move_left_dist = np.random.randint(5, 50)
        move_right_dist = np.random.randint(5, 50)
        image[:, move_left_dist:210, :] = image[:, 0:210-move_left_dist, :]
        label[:, move_left_dist:210] = label[:, 0:210-move_left_dist]

        image[:, 300:-move_right_dist, :] = image[:, 300+move_right_dist:, :]
        label[:, 300:-move_right_dist] = label[:, 300+move_right_dist:]

    return image, label


# merge two image
def augment_merge(image0, label0, image_files, label_files, idx, angle, crop_range):
    image0, label0 = augments(image0, label0, angle, crop_range)

    if np.random.rand() < random_rate:
        return image0, label0

    Height, Width, _ = image0.shape
    start_y, start_x = int(0.5*Height), int(0.5*Width)
    #
    idx_next = idx - 1
    if idx == 0:
        idx_next = idx + 1
    img_file, label_file = image_files[idx_next], label_files[idx_next]
    image1 = cv2.imread(img_file)
    label1 = cv2.imread(label_file, 0)
    image1, label1 = augments(image1, label1, angle, crop_range)
    # image1 = resize_image(image1, Width, 0, ret_params=False, mode=cv2.INTER_LINEAR)
    # label1 = resize_image(label1, Width, 0, ret_params=False, mode=cv2.INTER_LINEAR)
    #
    zeros = np.zeros([724, 724], dtype='uint8')
    random_H = random.randint(0, 724)
    zeros[0:random_H, :] = 1
    theter = random.randint(0, 180)
    zeros = Image.fromarray(zeros)  # OpenCV转换成PIL.Image格式
    im_rotate = zeros.rotate(theter, expand=False)
    im_rotate = np.array(im_rotate)
    H, W = im_rotate.shape
    center_x, center_y = int(0.5 * W), int(0.5 * H)
    crop_img = im_rotate[center_y - start_y: center_y + Height - start_y, center_x - start_x: center_x + Width - start_x]

    img0 = image0 * cv2.merge([crop_img, crop_img, crop_img])
    lab0 = label0 * crop_img
    crop_img = 1 - crop_img
    img1 = image1 * cv2.merge([crop_img, crop_img, crop_img])
    lab1 = label1 * crop_img
    img = img0 + img1
    label = lab0 + lab1

    return img, label


