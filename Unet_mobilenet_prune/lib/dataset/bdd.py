import numpy as np
import json
import os
from .AutoDriveDataset import AutoDriveDataset, AutoDriveDataset_2head, AutoDriveDataset_detect, AutoDriveDataset_segment
from .convert import convert, id_dict, id_dict_single, read_VOCxml, read_json
from tqdm import tqdm


single_cls = False       # just detect vehicle


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'images':, 'information':, ......}
        images: images path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for lane in tqdm(list(self.lane_list)):
            lane_path = str(lane)
            label_path = lane_path.replace(str(self.lane_root), str(self.label_root)).replace(".png", ".json")
            image_path = lane_path.replace(str(self.lane_root), str(self.img_root)).replace(".png", ".jpg")
            mask_path = lane_path.replace(str(self.lane_root), str(self.mask_root))

            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.filter_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                if category == "traffic light":
                    color = obj['attributes']['trafficLightColor']
                    category = "tl_" + color
                if category in id_dict.keys():
                    x1 = float(obj['box2d']['x1'])
                    y1 = float(obj['box2d']['y1'])
                    x2 = float(obj['box2d']['x2'])
                    y2 = float(obj['box2d']['y2'])
                    cls_id = id_dict[category]
                    if single_cls:
                         cls_id=0
                    gt[idx][0] = cls_id
                    box = convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)
                

            rec = [{
                'images': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass


class BddDataset_2head(AutoDriveDataset_2head):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'images':, 'information':, ......}
        images: images path
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for lane in tqdm(list(self.lane_list)):
            lane_path = str(lane)
            label_path = lane_path.replace(str(self.lane_root), str(self.label_root))
            image_path = lane_path.replace(str(self.lane_root), str(self.img_root)).replace(".png", ".jpg")
            if os.path.exists(label_path.replace(".png", ".json")):
                gt = read_json(label_path.replace(".png", ".json"), width, height)
            elif os.path.exists(label_path.replace(".png", ".xml")):
                gt = read_VOCxml(label_path.replace(".png", ".xml"), width, height)
            else:
                print('not find'+label_path)
                assert 0, 'not find .json or .xml file'

            # with open(label_path, 'r') as f:
            #     label = json.load(f)
            # data = label['frames'][0]['objects']
            # data = self.filter_data(data)
            # gt = np.zeros((len(data), 5))
            # for idx, obj in enumerate(data):
            #     category = obj['category']
            #     if category == "traffic light":
            #         color = obj['attributes']['trafficLightColor']
            #         category = "tl_" + color
            #     if category in id_dict.keys():
            #         x1 = float(obj['box2d']['x1'])
            #         y1 = float(obj['box2d']['y1'])
            #         x2 = float(obj['box2d']['x2'])
            #         y2 = float(obj['box2d']['y2'])
            #         cls_id = id_dict[category]
            #         if single_cls:
            #             cls_id = 0
            #         gt[idx][0] = cls_id
            #         box = convert((width, height), (x1, x2, y1, y2))
            #         gt[idx][1:] = list(box)

            rec = [{
                'images': image_path,
                'label': gt,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    # def filter_data(self, data):
    #     remain = []
    #     for obj in data:
    #         if 'box2d' in obj.keys():  # obj.has_key('box2d'):
    #             if single_cls:
    #                 if obj['category'] in id_dict_single.keys():
    #                     remain.append(obj)
    #             else:
    #                 remain.append(obj)
    #     return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """
        """
        pass


class BddDataset_detect(AutoDriveDataset_detect):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'images':, 'information':, ......}
        images: images path
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for img_path in tqdm(list(self.img_list)):
            img_path = str(img_path)
            label_path = img_path.replace(str(self.img_root), str(self.label_root))
            if os.path.exists(label_path.replace(".jpg", ".json")):
                gt = read_json(label_path.replace(".jpg", ".json"), width, height)
            elif os.path.exists(label_path.replace(".jpg", ".xml")):
                gt = read_VOCxml(label_path.replace(".jpg", ".xml"), width, height)
            else:
                print('not find'+label_path)
                assert 0, 'not find .json or .xml file'

            rec = [{
                'images': img_path,
                'label': gt,
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """
        """
        pass


class BddDataset_segment(AutoDriveDataset_segment):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'images':, 'information':, ......}
        images: images path
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for lane in tqdm(list(self.lane_list)):
            lane_path = str(lane)
            image_path = lane_path.replace(str(self.lane_root), str(self.img_root)).replace(".png", ".jpg")
            rec = [{
                'images': image_path,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """
        """
        pass


