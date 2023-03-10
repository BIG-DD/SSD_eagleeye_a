import torch
import logging
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
from lib.utils.util import initialize_weights
from lib.models.common import Detect, Detect_head, Segment_head
from lib.models.yolov3 import yolov3_tiny
from lib.models.yolov4 import yolov4_tiny, yolov4_tiny_origin, yolov4_tiny_mask, yolov4_tiny_mask_advance, yolov4_tiny_mask_advance_1branch
from lib.models.mobilenetv2 import mobilenetV2_1_2head, mobilenetV2_half_2head, mobilenetV2_quarter_2head
from lib.models.yolov5 import YOLOV5S_2head, YOLOP_2head, YOLOV5l6_2head, YOLOP_3head, YOLOV5S_segment
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.config import cfg

num_det_class = cfg.num_det_class

Dict = {'yolov3_tiny': yolov3_tiny, 'yolov4_tiny': yolov4_tiny,
        'yolov4_tiny_origin': yolov4_tiny_origin, 'yolov4_tiny_mask': yolov4_tiny_mask,
        'yolov4_tiny_mask_advance': yolov4_tiny_mask_advance, 'yolov4_tiny_mask_advance_1branch':yolov4_tiny_mask_advance_1branch}



class MCnet_detect_head(nn.Module):
    def __init__(self, num_det_class, block_cfg):
        super(MCnet_detect_head, self).__init__()
        layers, save = [], []
        self.nc = num_det_class  # traffic or not
        self.gr = 1.0
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            # print(i)
            block = eval(block) if isinstance(block, str) else block  # eval strings
            # print(block)
            if block is Detect_head:
                self.detector_index = i

            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        assert self.detector_index == self.det_out_idx

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect_head):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        det_out = None
        for i, block in enumerate(self.model):
            # if i > 6:
            # print(x)
            # print(block)
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]  # calculate concat detect
            x = block(x)
            if i == self.detector_index:  # save detect result
                det_out = x
            cache.append(x if block.index in self.save else None)
            # if isinstance(block, ConvBNReLU_mask):
            #     print(x)

        return det_out  # det

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m0 = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.conv.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 images)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            # mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


class MCnet_segment_head(nn.Module):
    def __init__(self, num_det_class, block_cfg):
        super(MCnet_segment_head, self).__init__()
        layers, save = [], []
        self.nc = num_det_class  # traffic or not
        self.gr = 1.0
        self.segment_index = -1
        self.seg_out_idx = block_cfg[0][0]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            # print(i)
            block = eval(block) if isinstance(block, str) else block  # eval strings
            # print(block)
            if block is Segment_head:
                self.segment_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        assert self.segment_index == self.seg_out_idx

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        initialize_weights(self)

    def forward(self, x):
        cache = []
        ll_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]  # calculate concat detect
            # print(block)
            x = block(x)
            if i == self.seg_out_idx:  # save segment result
                ll_out = x
                # ll_out = torch.sigmoid(x)
            cache.append(x if block.index in self.save else None)

        return ll_out  # det, ll

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.conv.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 images)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            # mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


class MCnet_2head(nn.Module):
    def __init__(self, num_det_class, block_cfg):
        super(MCnet_2head, self).__init__()
        layers, save = [], []
        self.nc = num_det_class  # traffic or not
        self.gr = 1.0
        self.detector_index = -1
        self.segment_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            # print(i)
            block = eval(block) if isinstance(block, str) else block  # eval strings
            # print(block)
            if block is Detect_head:
                self.detector_index = i
            if block is Segment_head:
                self.segment_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        assert self.detector_index == self.det_out_idx
        assert self.segment_index == self.seg_out_idx

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect_head):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        ll_out = None
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]  # calculate concat detect
            # print(block)
            # print(x.shape)
            x = block(x)
            if i == self.seg_out_idx:  # save segment result
                ll_out = x
                # ll_out = torch.sigmoid(x)
            if i == self.detector_index:  # save detect result
                det_out = x
            cache.append(x if block.index in self.save else None)

        return det_out, ll_out  # det, ll

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.conv.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 images)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            # mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


class MCnet_3head(nn.Module):
    def __init__(self, num_det_class, block_cfg):
        super(MCnet_3head, self).__init__()
        layers, save = [], []
        self.nc = num_det_class  # traffic or not
        self.gr = 1.0
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]

        # Build model
        # [-1, Focus, [3, 64, 3]]
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
                                                                             block.from_]  # calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:  # save driving area segment result
                out.append(torch.sigmoid(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0, det_out)
        return out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 images)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net_detect(cfg, logger):
    Dict = {'yolov3_tiny': yolov3_tiny, 'yolov4_tiny': yolov4_tiny,
            'yolov4_tiny_origin':yolov4_tiny_origin, 'yolov4_tiny_mask':yolov4_tiny_mask,
            'yolov4_tiny_mask_advance':yolov4_tiny_mask_advance,'yolov4_tiny_mask_advance_1branch':yolov4_tiny_mask_advance_1branch}
    m_block_cfg = None
    if Dict.get(cfg.MODEL.NAME):  # 查找key
        m_block_cfg = Dict[cfg.MODEL.NAME]
    else:
        assert (cfg.MODEL.NAME+': not found ')

    # m_block_cfg = YOLOV5S_2head
    # cfg.MODEL.NAME = 'YOLOV5S_2head'
    logger.info("=> m_block_cfg '{}'".format(cfg.MODEL.NAME))
    model = MCnet_detect_head(cfg.num_det_class, m_block_cfg)
    return model


def get_net_segment(cfg, logger):
    Dict = {'YOLOV5S_segment': YOLOV5S_segment}
    m_block_cfg = None
    if Dict.get(cfg.MODEL.NAME):  # 查找key
        m_block_cfg = Dict[cfg.MODEL.NAME]
    else:
        assert (cfg.MODEL.NAME+': not found ')

    # m_block_cfg = YOLOV5S_2head
    # cfg.MODEL.NAME = 'YOLOV5S_2head'
    logger.info("=> m_block_cfg '{}'".format(cfg.MODEL.NAME))
    model = MCnet_segment_head(cfg.num_det_class, m_block_cfg)
    return model


def get_net_2head(cfg, logger):
    Dict = {'YOLOV5S_2head': YOLOV5S_2head, 'YOLOV5l6_2head': YOLOV5l6_2head, 'YOLOP_2head': YOLOP_2head,
            'YOLOV5S_mobilenetV2_half_2head': mobilenetV2_half_2head, 'YOLOV5S_mobilenetV2_quarter_2head':mobilenetV2_quarter_2head,
            'YOLOV5S_mobilenetV2_1_2head': mobilenetV2_1_2head}
    m_block_cfg = None
    if Dict.get(cfg.MODEL.NAME):  # 查找key
        m_block_cfg = Dict[cfg.MODEL.NAME]
    else:
        assert (cfg.MODEL.NAME+': not found ')

    # m_block_cfg = YOLOV5S_2head
    # cfg.MODEL.NAME = 'YOLOV5S_2head'
    logger.info("=> m_block_cfg '{}'".format(cfg.MODEL.NAME))
    model = MCnet_2head(cfg.num_det_class, m_block_cfg)
    return model


def get_net_3head(cfg, logger):
    Dict = {'YOLOP_3head': YOLOP_3head}
    m_block_cfg = None
    if Dict.get(cfg.MODEL.NAME):  # 查找key
        m_block_cfg = Dict[cfg.MODEL.NAME]
    else:
        assert (cfg.MODEL.NAME+': not found ')

    # m_block_cfg = YOLOV5S_2head
    # cfg.MODEL.NAME = 'YOLOV5S_2head'
    logger.info("=> m_block_cfg '{}'".format(cfg.MODEL.NAME))
    model = MCnet_3head(cfg.num_det_class, m_block_cfg)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from torchsummary import summary

    logging.basicConfig(filename=str(),
                        format='')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    model = get_net_2head(cfg, logger)
    input_ = torch.randn((1, 3, 384, 384))
    gt_ = torch.rand((1, 2, 384, 384))
    metric = SegmentationMetric(2)
    detects, lane_line_seg = model(input_)
    # detects, dring_area_seg, lane_line_seg = model_out(input_)

    for det in detects:
        print(det.shape)

    print(lane_line_seg.shape)
 
    print(model)

    summary(model, (3, 384, 384), device="cpu")
