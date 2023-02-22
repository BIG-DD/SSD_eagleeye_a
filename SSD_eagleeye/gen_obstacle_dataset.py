import os, sys
from writeVOCxml import GEN_Annotations
import random
import numpy as np
import cv2
import copy
import shutil
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from threading import Thread
from multiprocessing import Pool


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

        boxs.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        labels.append(obj_name)

    return labels, boxs


def augment_flip():
    save_path = '/media/z590/D/DataSet/BYD_obstacle/DataCleaning_finish_merge/VOC2012_0921/VOC2012/'
    dir_path = '/media/z590/D/DataSet/BYD_obstacle/DataCleaning_finish_merge/VOC2012/'
    exts = ['xml']
    file_list = get_files(dir_path, exts)
    for i, file in enumerate(file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_list)))
        sys.stdout.flush()
        _, name = os.path.split(file)
        print(name)
        # copy
        shutil.copy(file, save_path+'Annotations/'+name)
        shutil.copy(file.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg'), save_path+'JPEGImages/'+name.replace('.xml', '.jpg'))
        # flip
        img = cv2.imdecode(
            np.fromfile(file.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg'), dtype=np.uint8), -1)
        h, w, _ = img.shape
        xml_boxs = []
        xml_labels = []
        labels, boxes = read_VOCxml(file)
        is_save = False
        for j, box in enumerate(boxes):
            xml_boxs.append([w-box[0], box[1], w-box[2], box[3]])
            xml_labels.append(labels[j])
            if labels[j] in ['parking_no']:
                is_save = True
            if labels[j] in ['non_motor_vehicle', 'person', 'traffic_cone', 'ground_lock_open', 'ground_lock_close']:
                if int(w*0.15) < min(box[0], box[2]) and max(box[0], box[2]) < int(w*0.85):
                    is_save = True

        if is_save:
            img1 = cv2.flip(img, 1)
            cv2.imwrite(save_path+'JPEGImages/'+name.replace('.xml', '_flip.jpg'), img1)
            creat_xml(save_path+'Annotations/', name.replace('.xml', '_flip.xml'), xml_labels, xml_boxs, w, h)


def compute_Union_rect(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross


def gen_merge_picture(img_gray, ori_gray, ori_img, ori_png, crop_img, img, box):
    b = int(img_gray - ori_gray)  # b 是亮度
    h, w, ch = ori_img.shape
    blank = np.zeros([h, w, ch], ori_img.dtype)
    new_img = cv2.addWeighted(ori_img, 1, blank, 0, b)  # 改变像素的API
    # ******************* modify gray val

    # 修改char_img的RGB值
    start_x, start_y, end_x, end_y = 10000, 10000, 0, 0
    for row in range(ori_img.shape[0]):
        for col in range(ori_img.shape[1]):
            if ori_png[row, col] == 0:
                continue
            if start_y == 10000: start_y = row
            end_y = row
            # ******************* modify gray val
            crop_img[row, col, 0] = new_img[row, col, 0]
            crop_img[row, col, 1] = new_img[row, col, 1]
            crop_img[row, col, 2] = new_img[row, col, 2]

    for col in range(ori_img.shape[1]):
        for row in range(ori_img.shape[0]):
            if ori_png[row, col] == 0:
                continue
            if start_x == 10000: start_x = col
            end_x = col

    img[box[1]:box[3], box[0]:box[2]] = crop_img
    new_box = [box[0] + start_x, box[1] + start_y, box[0] + end_x, box[1] + end_y]
    # print(box)

    return img, new_box


def calculate_candidates(part_pickup_boxes, part_pickup_imgs, index, box, ori_b, ori_g, ori_r, ori_gray, histequ_b, histequ_g, histequ_r, histequ_gray):
    candidates0, candidates1 = [], []
    length = len(part_pickup_boxes)
    union_rect = 0
    for i in range(length):
        pickup_box = part_pickup_boxes[i]
        # for p_box in pickup_box:
        union_rect += compute_Union_rect(box, pickup_box)

    if union_rect < 20:
        img = part_pickup_imgs
        img = cv2.resize(img, (1280, 720))
        crop_img = img[box[1]:box[3], box[0]:box[2]]

        # R,G,B mean val
        ori_img = copy.deepcopy(crop_img)
        # cv2.imshow('ori_img2', ori_img)
        # cv2.waitKey()
        B, G, R = cv2.split(ori_img)
        b = B.ravel()[np.flatnonzero(B)]  # 非零的值
        avg_b = sum(b) / len(b)  # 计算均值
        g = G.ravel()[np.flatnonzero(G)]
        avg_g = sum(g) / len(g)
        r = R.ravel()[np.flatnonzero(R)]
        avg_r = sum(r) / len(r)

        # ******************* modify gray val
        crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        crop_img_gray = np.mean(crop_img_gray[:])
        mean0 = abs(np.mean([abs(ori_b - avg_b), abs(ori_g - avg_g), abs(ori_r - avg_r)]))
        mean1 = abs(np.mean([abs(histequ_b - avg_b), abs(histequ_g - avg_g), abs(histequ_r - avg_r)]))
        std0 = np.var([ori_b - avg_b, ori_g - avg_g, ori_r - avg_r])
        std1 = np.var([histequ_b - avg_b, histequ_g - avg_g, histequ_r - avg_r])
        if mean0 < mean1:
            candidates0.append([index, crop_img_gray, abs(crop_img_gray - ori_gray), mean0, std0])
        else:
            candidates1.append([index, crop_img_gray, abs(crop_img_gray - histequ_gray), mean0, std1])
    return [candidates0, candidates1]


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def compute_threads(part_pickup_boxes, part_pickup_imgs, index, box, ori_b, ori_g, ori_r, ori_gray, histequ_b, histequ_g, histequ_r, histequ_gray):
    length = len(part_pickup_boxes)
    threads_num = 20
    results0 = []
    results1= []
    for i in range(int(length/threads_num)):
        threads = []
        for j in range(threads_num):
            start_index = i*threads_num + j
            th = MyThread(calculate_candidates, args=(part_pickup_boxes[start_index], part_pickup_imgs[start_index],
                                                      start_index+index, box, ori_b, ori_g, ori_r, ori_gray, histequ_b, histequ_g, histequ_r, histequ_gray))
            threads.append(th)
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        for th in threads:
            candi = th.get_result()
            if len(candi[0]) > 0:
                results0.append(candi[0])
            if len(candi[1]) > 0:
                results1.append(candi[1])
    return [results0, results1]


def compute_process(pickup_boxes, pickup_imgs,
                            box, ori_b, ori_g, ori_r, ori_gray, histequ_b, histequ_g, histequ_r, histequ_gray, processes_num=4):
    process_pool = Pool(processes=processes_num)
    results = []
    index_len = len(pickup_boxes)
    for k in range(processes_num):
        start_index = int(k * 1.0 / processes_num * index_len)
        end_index = int((k + 1) * 1.0 / processes_num * index_len)
        candidates = process_pool.apply_async(compute_threads, args=(pickup_boxes[start_index:end_index], pickup_imgs[start_index:end_index],
        start_index, box, ori_b, ori_g, ori_r, ori_gray, histequ_b, histequ_g, histequ_r, histequ_gray))

        results.append(candidates)
    process_pool.close()
    process_pool.join()

    candidates0, candidates1 = [], []
    for result in results:
        candidates00 = result.get()[0]
        candidates10 = result.get()[1]
        for i, can in enumerate(candidates00):
            candidates0.append(can[0])
        for i, can in enumerate(candidates10):
            candidates1.append(can[0])

    return candidates0, candidates1


def augment_cutmix():
    save_anno = '/media/z590/D/DataSet/BYD_obstacle/train_val/VOC2012_0921/imshow/'
    dir_path = '/media/z590/D/DataSet/BYD_obstacle/train_val/VOC2012_0921/VOC2012/Annotations/'
    exts = ['xml']
    file_lists = get_files(dir_path, exts)
    random.shuffle(file_lists)
    pickup_labels, pickup_boxes, pickup_imgs, is_save = [], [], [], []
    for i, pickup_file in enumerate(file_lists):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_lists)))
        sys.stdout.flush()
        if i >= 10000:
            continue
        pickup_label, pickup_box = read_VOCxml(pickup_file)
        pickup_labels.append(pickup_label)
        is_save.append(False)
        img = cv2.imdecode(
            np.fromfile(pickup_file.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg'),
                        dtype=np.uint8), -1)
        img = cv2.resize(img, (640, 360))
        pickup_boxes.append(pickup_box)
        pickup_imgs.append(img)

    # read crop label
    crop_path = '/media/z590/D/DataSet/BYD_obstacle/DataCleaning_finish_merge/crop_label_finish/'
    exts = ['png']
    png_lists = get_files(crop_path, exts)
    random.shuffle(png_lists)
    for i, file in enumerate(png_lists):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(png_lists)))
        crop_ori_png = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
        crop_ori_png = crop_ori_png[:, :, 3]
        crop_ori_img = cv2.imdecode(np.fromfile(file.replace('.png', '.jpg'), dtype=np.uint8), -1)
        # R,G,B mean val
        ori_img = copy.deepcopy(crop_ori_img)
        ori_img[crop_ori_png != 0] = (0,0,0)
        # cv2.imshow('ori_img0', ori_img)
        B, G, R = cv2.split(ori_img)
        b = B.ravel()[np.flatnonzero(B)]  # 非零的值
        ori_b = sum(b) / len(b)  # 计算均值
        g = G.ravel()[np.flatnonzero(G)]
        ori_g = sum(g) / len(g)
        # 色调（H）
        r = R.ravel()[np.flatnonzero(R)]
        ori_r = sum(r) / len(r)

        crop_ori_gray = cv2.cvtColor(crop_ori_img, cv2.COLOR_BGR2GRAY)
        ori_gray = np.mean(crop_ori_gray[:])
        # HistEqu
        b, g, r = cv2.split(crop_ori_img)
        # 创建局部直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        # 对每一个通道进行局部直方图均衡化
        b, g, r = clahe.apply(b), clahe.apply(g), clahe.apply(r)
        # 合并处理后的三通道 成为处理后的图
        crop_histequ = cv2.merge([b, g, r])
        # R,G,B mean val
        ori_img = copy.deepcopy(crop_histequ)
        ori_img[crop_ori_png != 0] = (0, 0, 0)
        # cv2.imshow('ori_img1', ori_img)
        B, G, R = cv2.split(ori_img)
        b = B.ravel()[np.flatnonzero(B)]  # 非零的值
        histequ_b = sum(b) / len(b)  # 计算均值
        g = G.ravel()[np.flatnonzero(G)]
        histequ_g = sum(g) / len(g)
        # 色调（H）
        r = R.ravel()[np.flatnonzero(R)]
        histequ_r = sum(r) / len(r)

        crop_histequ_gray = cv2.cvtColor(crop_histequ, cv2.COLOR_BGR2GRAY)
        histequ_gray = np.mean(crop_histequ_gray[:])

        path, name = os.path.split(file)
        name = name.replace('.png', '')
        index = name.rfind('_')
        new_name = name[: index]+'.xml'
        label_index = int(name[index+1:])
        # label = path[path.rfind('/')+1:]
        # print(index)
        xml_file = dir_path+new_name
        ori_img = cv2.imdecode(
            np.fromfile(xml_file.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg'), dtype=np.uint8), -1)
        H, W, _ = ori_img.shape

        labels, boxes = read_VOCxml(xml_file)
        label = labels[label_index]
        box = boxes[label_index]
        center_x = int(0.5*(box[2] + box[0]))
        center_y = int(0.5*(box[3] + box[1]))
        box[0] = max(0, int(center_x - 0.5*crop_ori_png.shape[1]))
        box[1] = max(0, int(center_y - 0.5*crop_ori_png.shape[0]))
        box[2] = min(W, box[0]+crop_ori_png.shape[1])
        box[3] = min(H, box[1]+crop_ori_png.shape[0])
        count = 0

        if label == 'traffic_cone' :  # 5
            count = 6
        if label == 'ground_lock_open':  # 15
            count = 16
        if label == 'ground_lock_close':  # 100
            count = 100

        candidates0, candidates1 = compute_process(pickup_boxes, pickup_imgs,
                                box, ori_b, ori_g, ori_r, ori_gray, histequ_b, histequ_g, histequ_r, histequ_gray, processes_num=14)

        # for j, pickup_file in enumerate(file_lists):
        #     sys.stdout.write('\r>> Converting image %d/%d->%d/%d' % (i + 1, len(png_lists), j, len(file_lists)))
        #     sys.stdout.flush()
        #     if j > 40000:
        #         continue
        #     pickup_box = pickup_boxes[j]
        #     union_rect = 0
        #     for p_box in pickup_box:
        #         union_rect += compute_Union_rect(box, p_box)
        #
        #     if union_rect < 20:
        #         img = pickup_imgs[j]
        #         img = cv2.resize(img, (1280, 720))
        #         crop_img = img[box[1]:box[3], box[0]:box[2]]
        #
        #         # R,G,B mean val
        #         ori_img = copy.deepcopy(crop_img)
        #         # cv2.imshow('ori_img2', ori_img)
        #         # cv2.waitKey()
        #         B, G, R = cv2.split(ori_img)
        #         b = B.ravel()[np.flatnonzero(B)]  # 非零的值
        #         avg_b = sum(b) / len(b)  # 计算均值
        #         g = G.ravel()[np.flatnonzero(G)]
        #         avg_g = sum(g) / len(g)
        #         r = R.ravel()[np.flatnonzero(R)]
        #         avg_r = sum(r) / len(r)
        #
        #         # ******************* modify gray val
        #         crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        #         crop_img_gray = np.mean(crop_img_gray[:])
        #         mean0 = abs(np.mean([abs(ori_b - avg_b), abs(ori_g - avg_g), abs(ori_r - avg_r)]))
        #         mean1 = abs(np.mean([abs(histequ_b - avg_b), abs(histequ_g - avg_g), abs(histequ_r - avg_r)]))
        #         std0 = np.var([ori_b - avg_b, ori_g - avg_g, ori_r - avg_r])
        #         std1 = np.var([histequ_b - avg_b, histequ_g - avg_g, histequ_r - avg_r])
        #         if mean0 < mean1:
        #             candidates0.append([j, crop_img_gray, abs(crop_img_gray - crop_ori_gray), mean0, std0])
        #         else:
        #             candidates1.append([j, crop_img_gray, abs(crop_img_gray - crop_histequ_gray), mean0, std1])

        random.shuffle(candidates0)
        candidates0 = sorted(candidates0, key=lambda x: (x[3], x[4]))

        for j, (index, gray, _, _, _) in enumerate(candidates0):
            if j > 0.5*count:
                continue
            is_save[index] = True
            pickup_labels[index].append(label)
            img = pickup_imgs[index]
            img = cv2.resize(img, (1280, 720))
            crop_img = img[box[1]:box[3], box[0]:box[2]]
            new_img, new_box = gen_merge_picture(gray, ori_gray, crop_ori_img, crop_ori_png, crop_img, img, box)
            new_img = cv2.resize(new_img, (640, 360))

            pickup_imgs[index] = new_img
            pickup_boxes[index].append(new_box)

        random.shuffle(candidates1)
        candidates1 = sorted(candidates1, key=lambda x: (x[3], x[4]))
        for j, (index, gray, _, _, _) in enumerate(candidates1):
            if j > 0.5*count:
                continue
            is_save[index] = True
            pickup_labels[index].append(label)
            img = pickup_imgs[index]

            img = cv2.resize(img, (1280, 720))
            crop_img = img[box[1]:box[3], box[0]:box[2]]
            new_img, new_box = gen_merge_picture(gray, histequ_gray, crop_histequ, crop_ori_png, crop_img, img, box)
            new_img = cv2.resize(new_img, (640, 360))

            pickup_imgs[index] = new_img
            pickup_boxes[index].append(new_box)

    for i, pickup_file in enumerate(file_lists):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_lists)))
        sys.stdout.flush()
        if i >= 10000:
            continue
        if is_save[i]:
            if os.path.exists(pickup_file.replace('.xml', '_aug.xml')):
                pickup_file = pickup_file.replace('.xml', '_aug.xml')
            path, name = os.path.split(pickup_file)

            img = pickup_imgs[i]
            boxes = pickup_boxes[i]

            img = cv2.resize(img, (1280, 720))
            h, w, _ = img.shape
            labels = pickup_labels[i]
            cv2.imwrite(save_anno + name.replace('.xml', '.jpg'), img)
            creat_xml(save_anno, name, labels, boxes, w, h)

def Statistics():
    # 'motor_vehicle': 130279,
    # 'limit_rod': 80605,
    # 'column': 79554,
    # 'wheel': 77281,
    # 'pole': 22864,
    # 'non_motor_vehicle': 15428, flip
    # 'traffic_cone': 11242,add to 2W
    # 'person': 11109, flip
    # 'ground_lock_open': 2729, add to 1W
    # 'parking_no': 2499,flip
    # 'ground_lock_close': 249 add to 1W
    dir_path = '/media/z590/D/DataSet/BYD_obstacle/DataCleaning_finish_merge/VOC2012/Annotations/'
    exts = ['xml']
    file_list = get_files(dir_path, exts)
    classes = {}
    for i, file in enumerate(file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_list)))
        sys.stdout.flush()
        _, name = os.path.split(file)
        print(name)
        labels, boxes = read_VOCxml(file)
        for label in labels:
            if classes.get(label):  # 查找key
                classes[label] += 1
            else:
                classes.update({label: 1})  # 增加元素
    print(classes)


def test():
    save_path = '/media/z590/D/DataSet/BYD_obstacle/train_val/VOC2012_0921/pick/'
    dir_path = '/media/z590/D/DataSet/BYD_obstacle/train_val/VOC2012_0921/VOC2012/'
    exts = ['jpg']
    file_list = get_files(dir_path, exts)
    for i, file in enumerate(file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(file_list)))
        sys.stdout.flush()
        _, name = os.path.split(file)
        if '_aug' in name:
            shutil.move(file, save_path+name)

if __name__ == '__main__':
    # test()

    # augment_flip()    # 1
    augment_cutmix()    # 2
    # Statistics()      # 3

# origin                         # augment
# {'traffic_cone':      11239,    42873,
#  'column':            79554,    98359,
#  'person':            11109,    20075,
#  'motor_vehicle':     130279,   167877,
#  'wheel':             77281,    99510,
#  'pole':              22864,    29106,
#  'non_motor_vehicle': 15428,    29659,
#  'limit_rod':         80605,    105645,
#  'parking_no':        2499,     4998,
#  'ground_lock_open':  2729,     19614,
#  'ground_lock_close': 249}      7696

