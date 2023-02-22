import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from nets.mobilenetv1 import mobilenet_v1, mobilenet_v1_half
from nets.mobilenetv2 import mobilenet_v2, corner_point_mobilenetv2_4, corner_point_mobilenetv2_4_tiny, corner_point_mobilenetv2_4_advance, \
    MobileNetV2_4_2, MobileNetV2_4_2_test
from nets.xception import xception
from nets.vgg import vgg as add_vgg
from nets.extras import add_extras, L2Norm, MFR, F_SSD
from nets.JacintoNetV2 import JacintoNetV2_lite, JacintoNetV2_yihang, JacintoNetV2_nano
import onnx
import onnxruntime as ort
import onnxsim
import math


class SSD(nn.Module):
    def __init__(self, backbone, num_classes=5, quantization=False, BACKEND='default', fuse_model=False):
        super(SSD, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        layers, save = [], []
        if backbone == 'MobileNetV2_4_2':
            block_cfg = MobileNetV2_4_2
        elif backbone == 'MobileNetV2_4_2_test':
            block_cfg = MobileNetV2_4_2_test
        else:
            return
        # Build model
        indexes = block_cfg[0]
        self.loc_out_idx = indexes[::2]
        self.conf_out_idx = indexes[1::2]
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            args.append(quantization)
            block = eval(block) if isinstance(block, str) else block  # eval strings
            # print(block)
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self._init_weights()

    def forward(self, x):
        cache = []
        loc, conf = list(), list()
        for i, block in enumerate(self.model):
            # print(x)
            # print(block)
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]  # calculate concat detect
            x = block(x)
            if i in self.loc_out_idx:
                loc.append(x.permute(0, 2, 3, 1).contiguous())
            if i in self.conf_out_idx:
                conf.append(x.permute(0, 2, 3, 1).contiguous())

            cache.append(x if block.index in self.save else None)

        # -------------------------------------------------------------#
        #   进行reshape方便堆叠
        # -------------------------------------------------------------#
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # -------------------------------------------------------------#
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        # -------------------------------------------------------------#
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )

        return output  # det

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.normal_(tensor, mean=0, std=1)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0)
                # m.weight.data.fill_(0.5)
                # m.bias.data.zero_()
            if isinstance(m, nn.Conv2d) and m.groups != 1 and m.kernel_size == (1, 1):
                nn.init.constant_(m.weight, 1.)


class SSD_Prune(nn.Module):
    def __init__(self, net_params, block_cfg, num_classes):
        super(SSD_Prune, self).__init__()
        self.num_classes = num_classes
        layers, save = [], []
        # Build model
        indexes = block_cfg[0]
        self.loc_out_idx = indexes[::2]
        self.conf_out_idx = indexes[1::2]
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block_params = net_params[i]
            # print(i)
            block = eval(block) if isinstance(block, str) else block  # eval strings
            # print(block)
            if len(block_params) != 0:
                block_ = block(*block_params)
            else:
                block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        self.model, self.save = nn.Sequential(*layers), sorted(save)

    def forward(self, x):
        cache = []
        loc, conf = list(), list()
        for i, block in enumerate(self.model):
            # print(x)
            # print(block)
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else \
                    [x if j == -1 else cache[j] for j in block.from_]  # calculate concat detect
            x = block(x)
            if i in self.loc_out_idx:
                loc.append(x.permute(0, 2, 3, 1).contiguous())
            if i in self.conf_out_idx:
                conf.append(x.permute(0, 2, 3, 1).contiguous())

            cache.append(x if block.index in self.save else None)

        # -------------------------------------------------------------#
        #   进行reshape方便堆叠
        # -------------------------------------------------------------#
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # -------------------------------------------------------------#
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        # -------------------------------------------------------------#
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )

        return output


class SSD300(nn.Module):
    def __init__(self, num_classes, backbone_name, pretrained = False):
        super(SSD300, self).__init__()
        self.num_classes    = num_classes
        if backbone_name    == "vgg":
            self.vgg        = add_vgg(pretrained)
            self.extras     = add_extras(1024, backbone_name)
            self.L2Norm     = L2Norm(512, 20)
            mbox            = [4, 6, 6, 6, 4, 4]
            
            loc_layers      = []
            conf_layers     = []
            backbone_source = [21, -2]
            #---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第21层和-2层可以用来进行回归预测和分类预测。
            #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
            #---------------------------------------------------#
            for k, v in enumerate(backbone_source):
                loc_layers  += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            #-------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            #-------------------------------------------------------------#  
            for k, v in enumerate(self.extras[1::2], 2):
                loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'mobilenetv1_half':
            self.mobilenet = mobilenet_v1_half(pretrained).features
            self.extras = add_extras(512, backbone_name)
            mbox = [6, 6, 6, 6, 4]

            loc_layers = []
            conf_layers = []
            self.backbone_source = [11, 13, 14, 15, 16]
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=1, padding=0)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=1, padding=0)]
        elif backbone_name == 'mobilenetv1':
            self.mobilenet = mobilenet_v1(pretrained).features
            self.extras = add_extras(512, backbone_name)
            mbox = [6, 6, 6, 6, 4]
            loc_layers = []
            conf_layers = []
            self.backbone_source = [11, 13, 14, 15, 16]
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=1, padding=0)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=1, padding=0)]
        elif backbone_name == 'mobilenetv1_F_SSD':
            self.mobilenet = mobilenet_v1(pretrained).features
            self.extras = add_extras(512, backbone_name)
            self.F_SSD = F_SSD(128, 512)
            mbox = [6, 6, 6, 6, 4]
            loc_layers = []
            conf_layers = []
            self.backbone_source = [5, 11, 13, 14, 15, 16]
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=1, padding=0)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=1, padding=0)]
        elif backbone_name == 'mobilenetv1_MFR':
            self.mobilenet = mobilenet_v1(pretrained).features
            self.extras = add_extras(512, backbone_name)
            self.MFR = MFR(512)
            mbox = [6, 6, 6, 6, 4]

            loc_layers = []
            conf_layers = []
            self.backbone_source = [12, 13, 14, 15, 16]
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=1, padding=0)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=1, padding=0)]
        elif backbone_name == 'xception':
            self.xception = xception(pretrained).features
            self.extras = add_extras(512, backbone_name)
            mbox = [6, 6, 6, 6, 4]
            loc_layers = []
            conf_layers = []
            self.backbone_source = [11-4, 13-4, 15-4, 16-4, 17-4]
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=1, padding=0)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=1, padding=0)]
        elif backbone_name == 'mobilenetv2':
            self.mobilenet  = mobilenet_v2(pretrained).features
            self.extras     = add_extras(1280, backbone_name)
            mbox            = [6, 6, 6, 6, 6, 6]

            loc_layers      = []
            conf_layers     = []
            backbone_source = [13, -1]
            for k, v in enumerate(backbone_source):
                loc_layers  += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'mobilenetv2_half':
            self.mobilenet  = mobilenet_v2(pretrained, compress=0.5).features
            self.extras     = add_extras(640, backbone_name)
            mbox            = [6, 6, 6, 6, 4]

            loc_layers      = []
            conf_layers     = []
            backbone_source = [13, -1]
            for k, v in enumerate(backbone_source):
                loc_layers  += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                    loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'mobilenetv2_quarter':
            self.mobilenet  = mobilenet_v2(pretrained, compress=0.25).features
            self.extras     = add_extras(320, backbone_name)
            mbox            = [4, 6, 6, 6, 4]

            loc_layers      = []
            conf_layers     = []
            backbone_source = [13, -1]
            for k, v in enumerate(backbone_source):
                loc_layers  += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                    loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'JacintoNetV2_lite':
            self.backbone = JacintoNetV2_lite(pretrained)
            self.extras = add_extras(512, backbone_name)
            mbox = [6, 6, 6, 6, 4]

            loc_layers = []
            conf_layers = []
            self.backbone_source = [10, 13, 14, 15, 16]  # 获得的特征层的输出, 用来进行回归预测和分类预测。
            # ---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第21层和-2层可以用来进行回归预测和分类预测。
            #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
            # ---------------------------------------------------#
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v[0].out_channels, mbox[k] * 4, kernel_size=1, padding=0)]
                conf_layers += [nn.Conv2d(v[0].out_channels, mbox[k] * num_classes, kernel_size=1, padding=0)]
        elif backbone_name == 'JacintoNetV2_yihang':
            self.backbone = JacintoNetV2_yihang(pretrained)
            self.extras = add_extras(512, backbone_name)
            mbox = [4, 6]
            loc_layers = []
            conf_layers = []
            self.backbone_source = [7, 10]  # 获得的特征层的输出, 用来进行回归预测和分类预测。
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v[0].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v[0].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'JacintoNetV2_nano':
            self.backbone = JacintoNetV2_nano(pretrained)
            self.extras = add_extras(512, backbone_name)
            mbox = [2, 2]
            loc_layers = []
            conf_layers = []
            self.backbone_source = [7, 10]  # 获得的特征层的输出, 用来进行回归预测和分类预测。
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v[0].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v[0].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'corner_point_mobilenetv2_4_tiny':
            self.backbone = corner_point_mobilenetv2_4_tiny(pretrained).features
            self.extras = add_extras(512, backbone_name)
            mbox = [4, 4]
            loc_layers = []
            conf_layers = []
            self.backbone_source = [10, 14]  # 获得的特征层的输出, 用来进行回归预测和分类预测。
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'corner_point_mobilenetv2_4':
            self.backbone = corner_point_mobilenetv2_4(pretrained).features
            self.extras = add_extras(512, backbone_name)
            mbox = [4, 4]
            loc_layers = []
            conf_layers = []
            self.backbone_source = [10, 14]  # 获得的特征层的输出, 用来进行回归预测和分类预测。
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'corner_point_mobilenetv2_4_advance':
            self.backbone = corner_point_mobilenetv2_4_advance(pretrained).features
            self.extras = add_extras(512, backbone_name)
            mbox = [4, 4]
            loc_layers = []
            conf_layers = []
            self.backbone_source = [14, 17]  # 获得的特征层的输出, 用来进行回归预测和分类预测。
            for k, v in enumerate(self.extras):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        else:
            print('ERROR')

        self.loc            = nn.ModuleList(loc_layers)
        self.conf           = nn.ModuleList(conf_layers)
        self.backbone_name  = backbone_name
        
    def forward(self, x):
        #---------------------------#
        #   x是300,300,3
        #---------------------------#
        sources = list()
        loc     = list()
        conf    = list()

        #---------------------------#
        #   获得conv4_3的内容
        #   shape为38,38,512
        #---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23):
                x = self.vgg[k](x)
            sources.append(x)
            # ---------------------------#
            #   获得conv7的内容
            #   shape为19,19,1024
            # ---------------------------#
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)

            sources.append(x)
            # -------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            # -------------------------------------------------------------#
            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    sources.append(x)
        elif self.backbone_name == 'mobilenetv1_half':
            count = 0
            for k in range(len(self.mobilenet)):
                x = self.mobilenet[k](x)
                if k in self.backbone_source:
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
        elif self.backbone_name == 'mobilenetv1':
            count = 0
            for k in range(len(self.mobilenet)):
                x = self.mobilenet[k](x)
                if k in self.backbone_source:
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
        elif self.backbone_name == 'xception':
            count = 0
            for k in range(len(self.xception)):
                x = self.xception[k](x)
                if k in self.backbone_source:
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
        elif self.backbone_name == 'mobilenetv1_MFR':
            count = 0
            for k in range(len(self.mobilenet)):
                x = self.mobilenet[k](x)
                if k in self.backbone_source:
                    if k == 12:
                        out = x
                        x = self.MFR(front, x)
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
                    if k == 12:
                        x = out
                front = x
        elif self.backbone_name == 'mobilenetv1_F_SSD':
            count = 0
            for k in range(len(self.mobilenet)):
                x = self.mobilenet[k](x)
                if k in self.backbone_source:
                    if k == 5:
                        front = x
                    elif k == 11:
                        mid = x
                    elif k == 13:
                        out = self.F_SSD(front, mid, x)
                        v = self.extras[count]
                        sources.append(v(out))
                        count += 1
                        v = self.extras[count]
                        sources.append(v(x))
                        count += 1
                    else:
                        v = self.extras[count]
                        sources.append(v(x))
                        count += 1
        elif self.backbone_name == 'mobilenetv2':
            for k in range(14):
                x = self.mobilenet[k](x)
            # ---------------------------#
            #   conv4_3的内容
            # ---------------------------#
            sources.append(x)
            # ---------------------------#
            #   获得conv7的内容
            #   shape为19,19,1024
            # ---------------------------#
            for k in range(14, len(self.mobilenet)):
                x = self.mobilenet[k](x)
            sources.append(x)
            # -------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            # -------------------------------------------------------------#
            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                sources.append(x)
        elif self.backbone_name == 'mobilenetv2_half':
            for k in range(len(self.mobilenet)):
                x = self.mobilenet[k](x)
                if k == 13:
                    sources.append(x)
            sources.append(x)
            # -------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            # -------------------------------------------------------------#
            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                sources.append(x)
        elif self.backbone_name == 'mobilenetv2_quarter':
            for k in range(len(self.mobilenet)):
                x = self.mobilenet[k](x)
                if k == 13:
                    sources.append(x)
            sources.append(x)

            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                sources.append(x)
        elif self.backbone_name == 'JacintoNetV2_lite':
            count = 0
            for k in range(len(self.backbone)):
                x = self.backbone[k](x)
                if k in self.backbone_source:
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
        elif self.backbone_name == 'JacintoNetV2_yihang':
            count = 0
            for k in range(len(self.backbone)):
                x = self.backbone[k](x)
                if k in self.backbone_source:
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
        elif self.backbone_name == 'JacintoNetV2_nano':
            count = 0
            for k in range(len(self.backbone)):
                x = self.backbone[k](x)
                if k in self.backbone_source:
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
        elif self.backbone_name == 'corner_point_mobilenetv2_4_tiny':
            count = 0
            for k in range(len(self.backbone)):
                if k == 13:
                    x = self.backbone[k](out1, out2)
                else:
                    x = self.backbone[k](x)
                if k == 6:
                    out1 = x
                if k == 12:
                    out2 = x
                if k in self.backbone_source:
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
        elif self.backbone_name == 'corner_point_mobilenetv2_4':
            count = 0
            for k in range(len(self.backbone)):
                if k == 13:
                    x = self.backbone[k](out1, out2)
                else:
                    x = self.backbone[k](x)
                if k == 6:
                    out1 = x
                if k == 12:
                    out2 = x
                if k in self.backbone_source:
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
        elif self.backbone_name == 'corner_point_mobilenetv2_4_advance':
            count = 0
            for k in range(len(self.backbone)):
                if k == 13:
                    x = self.backbone[k](out1, out2)
                elif k == 16:
                    x = self.backbone[k](out3, out4)
                else:
                    x = self.backbone[k](x)
                if k == 6:out1 = x
                if k == 12:out2 = x
                if k == 10:out3 = x
                if k == 15:out4 = x
                if k in self.backbone_source:
                    v = self.extras[count]
                    sources.append(v(x))
                    count += 1
        else:
            print('ERROR')
        #-------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        #-------------------------------------------------------------#      
        for (x, l, c) in zip(sources, self.loc, self.conf):
            if torch.onnx.is_in_onnx_export():
                loc.append(l(x).view(l(x).size(0), -1))
                conf.append(c(x).view(c(x).size(0), -1))
            else:
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        #-------------------------------------------------------------#
        #   进行reshape方便堆叠
        #-------------------------------------------------------------#  
        loc     = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf    = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #-------------------------------------------------------------#
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        #-------------------------------------------------------------#     
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output


if __name__ == "__main__":
    # model = SSD(backbone='MobileNetV2_4_2', num_classes=5)
    # print(model)
    # # model = UNet(backbone="mobilenetv2", num_classes=2, pretrained_backbone=None)
    # # print(model)
    # from torchsummary import summary
    #
    # # model = mobilenetv2_to_mobilenetv1(model)
    #
    # model = model.to("cuda")
    # # flops = compute_conv_flops(model, cuda=True)
    # # print('flops: {}'.format(flops))
    #
    # summary(model, (3, 256, 256))

    num_classes = 6
    backbone = 'corner_point_mobilenetv2_4_tiny'
    pretrained = False
    model = SSD300(num_classes, backbone, pretrained)
    from eagleeye_pruning import compute_conv_flops
    flops = compute_conv_flops(model, cuda=True)
    print(model)
    # torch.save(model.state_dict(), '/media/z590/G/Pytorch/ssd-pytorch-master/logs/mobilenetv1_F_SSD.pth')

    from torchsummary import summary


    model = model.to("cuda")
    summary(model, (3, 256, 256))

    # Input
    inputs = torch.randn(1, 3, 512, 512)
    # with torch.no_grad():
    #     output = model(inputs)
    # print(output)
    model = model.to("cpu")
    onnx_path = '/media/z590/G/Pytorch/ssd-pytorch-master/logs/mobilenetv1_MFR.onnx'
    torch.onnx.export(model,
                      inputs,
                      onnx_path,
                      verbose=False,
                      opset_version=9,
                      input_names=['images'],
                      output_names=['loc', 'conf'])
    print('convert', onnx_path, 'to onnx finish!!!')

    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    print(onnx.helper.printable_graph(model_onnx.graph))  # print

    print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
    model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
    assert check, 'assert check failed'
    onnx.save(model_onnx, onnx_path)

    x = inputs.cpu().numpy()
    try:
        sess = ort.InferenceSession(onnx_path)

        for ii in sess.get_inputs():
            print("Input: ", ii)
        for oo in sess.get_outputs():
            print("Output: ", oo)

        print('read onnx using onnxruntime sucess')
    except Exception as e:
        print('read failed')
        raise e

#
# ======================JacintoNetV2_lite====50ms======================================
# Total params: 1,233,668
# Trainable params: 1,233,668
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.00
# Forward/backward pass size (MB): 139.70
# Params size (MB): 4.71
# Estimated Total Size (MB): 147.40
# ----------------------------------------------------------------

# =========================mobilenetv2_lite=======================================
# Total params: 2,810,634
# Trainable params: 2,810,634
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.00
# Forward/backward pass size (MB): 793.18
# Params size (MB): 10.72
# Estimated Total Size (MB): 806.90
# ---------------------------------------


# =========================mobilenetv2=======================================
# Total params: 5,139,172
# Trainable params: 5,139,172
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.00
# Forward/backward pass size (MB): 802.67
# Params size (MB): 19.60
# Estimated Total Size (MB): 825.27
# ----------------------------------------------------------------

# ============================mobilenetv2_half====================================
# Total params: 2,318,330
# Trainable params: 2,318,330
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.00
# Forward/backward pass size (MB): 482.61
# Params size (MB): 8.84
# Estimated Total Size (MB): 494.45
# ----------------------------------------------------------------

# ========================mobilenetv2_half_4========================================
# Total params: 2,158,986
# Trainable params: 2,158,986
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.00
# Forward/backward pass size (MB): 368.42
# Params size (MB): 8.24
# Estimated Total Size (MB): 379.66
# ----------------------------------------------------------------

# ===========================mobilenetv2_quarter=====================================
# Total params: 957,082
# Trainable params: 957,082
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.00
# Forward/backward pass size (MB): 340.71
# Params size (MB): 3.65
# Estimated Total Size (MB): 347.36
# ----------------------------------------------------------------

# ===========================mobilenetv1=====================================
# Total params: 2,272,892
# Trainable params: 2,272,892
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.00
# Forward/backward pass size (MB): 328.86
# Params size (MB): 8.67
# Estimated Total Size (MB): 340.53
# ----------------------------------------------------------------

# =============================mobilenetv1 + MFR===================================
# Total params: 2,542,716
# Trainable params: 2,542,716
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.00
# Forward/backward pass size (MB): 342.36
# Params size (MB): 9.70
# Estimated Total Size (MB): 355.06
# ----------------------------------------------------------------

# =========================JacintoNetV2_nano=======================================
# Total params: 709,690
# Trainable params: 709,690
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 54.30
# Params size (MB): 2.71
# Estimated Total Size (MB): 57.76
# ----------------------------------------------------------------

# ===========================JacintoNetV2_yihang==34ms+5.8ms===================================
# Total params: 778,938
# Trainable params: 778,938
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 58.55
# Params size (MB): 2.97
# Estimated Total Size (MB): 62.28
# ----------------------------------------------------------------

# =======================corner_point_mobilenetv2_4==25ms+5.3ms=======================================
# Total params: 446,248
# Trainable params: 446,248
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 120.41
# Params size (MB): 1.70
# Estimated Total Size (MB): 122.87
# ----------------------------------------------------------------

# =======================corner_point_mobilenetv2_4_advance=========================================
# Total params: 497,704
# Trainable params: 497,704
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 122.79
# Params size (MB): 1.90
# Estimated Total Size (MB): 125.44
# ----------------------------------------------------------------


# =============================corner_point_mobilenetv2_4_tiny=21ms+5.3ms==================================
# Total params: 226,992
# Trainable params: 226,992
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 109.46
# Params size (MB): 0.87
# Estimated Total Size (MB): 111.08
# ----------------------------------------------------------------





