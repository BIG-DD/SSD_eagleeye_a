# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from functools import reduce
from base import BaseModel
from models.backbonds import MobileNetV2, ResNet
from models.util import pth2onnx
from models.RM_Prune import ConvBNReLU_mask, ConvBN_mask, ResBlock, Bottleneck_mask_RM, Concat, Conv2d
from models.RM_Prune import ConvBNReLU, ConvBN, Bottleneck_RM_prune, ResBlock_RM_prune, get_mode_4_threshold
from torch.nn import ConvTranspose2d, MaxPool2d
import copy

# ------------------------------------------------------------------------------
#   Decoder block
# ------------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_unit):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, padding=1, stride=2,
                                         groups=out_channels)
        self.block_unit = block_unit

    def forward(self, input, shortcut):
        x = self.conv(input)
        x = self.deconv(x)
        x = torch.cat([x, shortcut], dim=1)
        x = self.block_unit(x)
        return x


# ------------------------------------------------------------------------------
#   Class of UNet
# ------------------------------------------------------------------------------
class UNet(BaseModel):
    def __init__(self, backbone="resnet18", num_classes=2, pretrained_backbone=None):
        super(UNet, self).__init__()
        if backbone == 'mobilenetv2':
            alpha = 1.0
            expansion = 4
            self.backbone = MobileNetV2.MobileNetV2(alpha=alpha, expansion=expansion, num_classes=None)
            self._run_backbone = self._run_backbone_mobilenetv2
            # # Stage 1
            # channel1 = MobileNetV2._make_divisible(int(96*alpha), 8)
            # block_unit = MobileNetV2.InvertedResidual(2*channel1, channel1, 1, expansion)
            # self.decoder1 = DecoderBlock(self.backbone.last_channel, channel1, block_unit)
            # # Stage 2
            channel2 = MobileNetV2._make_divisible(int(32 * alpha), 8)
            # block_unit = MobileNetV2.InvertedResidual(2*channel2, channel2, 1, expansion)
            # self.decoder2 = DecoderBlock(channel1, channel2, block_unit)
            # Stage 3
            channel3 = MobileNetV2._make_divisible(int(24 * alpha), 8)
            block_unit = MobileNetV2.InvertedResidual(2 * channel3, channel3, 1, expansion)
            self.decoder3 = DecoderBlock(channel2, channel3, block_unit)
            # Stage 4
            channel4 = MobileNetV2._make_divisible(int(16 * alpha), 8)
            block_unit = MobileNetV2.InvertedResidual(2 * channel4, channel4, 1, expansion)
            self.decoder4 = DecoderBlock(channel3, channel4, block_unit)

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                n_layers = 18
            elif backbone == 'resnet34':
                n_layers = 34
            elif backbone == 'resnet50':
                n_layers = 50
            elif backbone == 'resnet101':
                n_layers = 101
            else:
                raise NotImplementedError
            filters = 64
            self.backbone = ResNet.get_resnet(n_layers, num_classes=None)
            self._run_backbone = self._run_backbone_resnet
            block = ResNet.BasicBlock if (n_layers == 18 or n_layers == 34) else ResNet.Bottleneck
            # # Stage 1
            # last_channel = 8*filters if (n_layers==18 or n_layers==34) else 32*filters
            # channel1 = 4*filters if (n_layers==18 or n_layers==34) else 16*filters
            # downsample = nn.Sequential(ResNet.conv1x1(2*channel1, channel1), nn.BatchNorm2d(channel1))
            # block_unit = block(2*channel1, int(channel1/block.expansion), 1, downsample)
            # self.decoder1 = DecoderBlock(last_channel, channel1, block_unit)
            # # Stage 2
            channel2 = 2 * filters if (n_layers == 18 or n_layers == 34) else 8 * filters
            # downsample = nn.Sequential(ResNet.conv1x1(2*channel2, channel2), nn.BatchNorm2d(channel2))
            # block_unit = block(2*channel2, int(channel2/block.expansion), 1, downsample)
            # self.decoder2 = DecoderBlock(channel1, channel2, block_unit)
            # Stage 3
            channel3 = filters if (n_layers == 18 or n_layers == 34) else 4 * filters
            downsample = nn.Sequential(ResNet.conv1x1(2 * channel3, channel3), nn.BatchNorm2d(channel3))
            block_unit = block(2 * channel3, int(channel3 / block.expansion), 1, downsample)
            self.decoder3 = DecoderBlock(channel2, channel3, block_unit)
            # Stage 4
            channel4 = filters
            downsample = nn.Sequential(ResNet.conv1x1(2 * channel4, channel4), nn.BatchNorm2d(channel4))
            block_unit = block(2 * channel4, int(channel4 / block.expansion), 1, downsample)
            self.decoder4 = DecoderBlock(channel3, channel4, block_unit)

        else:
            raise NotImplementedError

        self.conv_last = nn.Sequential(
            nn.Conv2d(channel4, 3, kernel_size=3, padding=1),
            nn.Conv2d(3, num_classes, kernel_size=3, padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, padding=1, stride=2, groups=num_classes)
        )

        # Initialize
        self._init_weights()
        if pretrained_backbone is not None:
            self.backbone._load_pretrained_model(pretrained_backbone)

    def forward(self, input):
        # x1, x2, x3, x4, x5 = self._run_backbone(input)
        x1, x2, x3 = self._run_backbone(input)
        # x = self.decoder1(x5, x4)	 # 20*20
        # x = self.decoder2(x, x3)	 # 40*40
        x = self.decoder3(x3, x2)  # 80*80
        x = self.decoder4(x, x1)  # 160*160
        x = self.conv_last(x)  # 160*160
        # x = F.interpolate(x, size=input.shape[-2:], mode='bilinear', align_corners=True)
        return x

    def _run_backbone_mobilenetv2(self, input):
        x = input
        # Stage1 160*160
        x = reduce(lambda x, n: self.backbone.features[n](x), list(range(0, 2)), x)
        x1 = x
        # Stage2 80*80
        x = reduce(lambda x, n: self.backbone.features[n](x), list(range(2, 4)), x)
        x2 = x
        # Stage3 40*40
        x = reduce(lambda x, n: self.backbone.features[n](x), list(range(4, 7)), x)
        x3 = x
        # # Stage4 20*20
        # x = reduce(lambda x, n: self.backbone.features[n](x), list(range(7,14)), x)
        # x4 = x
        # # Stage5 10*10
        # x5 = reduce(lambda x, n: self.backbone.features[n](x), list(range(14,19)), x)
        return x1, x2, x3  # , x4, x5

    def _run_backbone_resnet(self, input):
        # Stage1
        x1 = self.backbone.conv1(input)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        # Stage2
        x2 = self.backbone.maxpool(x1)
        x2 = self.backbone.layer1(x2)
        # Stage3
        x3 = self.backbone.layer2(x2)
        # # Stage4
        # x4 = self.backbone.layer3(x3)
        # # Stage5
        # x5 = self.backbone.layer4(x4)
        return x1, x2, x3  # , x4, x5

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class UNet_resnet18(BaseModel):
    def __init__(self, backbone="resnet18", num_classes=2, pretrained_backbone=None):
        super(UNet_resnet18, self).__init__()
        self.inplanes = 64
        ## -------------Encoder-------------
        self.conv_bn_mask_relu = ConvBNReLU_mask(3, self.inplanes, kernel=3, stride=2, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.layer1 = self._make_layer(block, 1 * self.inplanes, num_layers=2, stride=strides[0])
        self.layer1_1 = Bottleneck_mask_RM(self.inplanes, self.inplanes)
        self.layer1_2 = Bottleneck_mask_RM(self.inplanes, self.inplanes)
        # self.layer2 = self._make_layer(block, 2 * self.inplanes, num_layers=2, stride=strides[1])
        self.layer2_1 = ResBlock(self.inplanes, 2 * self.inplanes, stride=2)
        self.layer2_2 = Bottleneck_mask_RM(2 * self.inplanes, 2 * self.inplanes)
        ## -------------Decoder--------------
        # Stage 3
        # block_unit = ResBlock(2*inplanes, int(inplanes), 1)
        # self.decoder3 = DecoderBlock(2*inplanes, inplanes, block_unit)
        # self.decoder3_upsample = DecoderBlock_upsample(2*self.inplanes, self.inplanes)

        self.decoder3_conv = Conv2d(2 * self.inplanes, self.inplanes)
        self.decoder3_upsample = ConvTranspose2d(self.inplanes, self.inplanes, kernel_size=4, stride=2, padding=1,
                                                 groups=self.inplanes)
        self.decoder3_cat = Concat()
        self.decoder3_block_unit = ResBlock(2 * self.inplanes, self.inplanes, 1)
        # Stage 4
        # block_unit = ResBlock(2*inplanes, int(inplanes), 1)
        # self.decoder4 = DecoderBlock(inplanes, inplanes, block_unit)
        # self.decoder4_upsample = DecoderBlock_upsample(self.inplanes, self.inplanes)
        self.decoder4_conv = Conv2d(self.inplanes, self.inplanes)
        self.decoder4_upsample = ConvTranspose2d(self.inplanes, self.inplanes, kernel_size=4, stride=2, padding=1,
                                                 groups=self.inplanes)
        self.decoder4_cat = Concat()
        self.decoder4_block_unit = ResBlock(2 * self.inplanes, self.inplanes, 1)

        self.conv_bn_mask5 = ConvBN_mask(self.inplanes, 32, kernel=3, stride=1, padding=1)

        self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
        self.deconv7 = ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, groups=num_classes)

    # Initialize
    # self._init_weights()
    # if pretrained_backbone is not None:
    # 	self.backbone._load_pretrained_model(pretrained_backbone)

    def forward(self, input):
        x1 = self.conv_bn_mask_relu(input)
        # Stage2
        x2 = self.maxpool(x1)
        x2 = self.layer1_1(x2)
        x2 = self.layer1_2(x2)
        # Stage3
        x3 = self.layer2_1(x2)
        x3 = self.layer2_2(x3)

        # x = self.decoder3(x3, x2)	 # 80*80
        x = self.decoder3_conv(x3)
        x = self.decoder3_upsample(x)
        x = self.decoder3_cat(x, x2)
        x = self.decoder3_block_unit(x)

        # x = self.decoder4(x, x1)	 # 160*160
        x = self.decoder4_conv(x)
        x = self.decoder4_upsample(x)
        x = self.decoder4_cat(x, x1)
        x = self.decoder4_block_unit(x)

        x = self.conv_bn_mask5(x)
        x = self.conv6(x)
        x = self.deconv7(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_mask(self, prune_scale_rate):  # 稀疏化训练一定epoch后把小于阈值部分的权重置为0
        total = 0
        # 统计mask数量
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    size = m.weight.data.shape[0]
                    # print(size)
                    data = m.weight.data.abs().clone()
                    data = data.to('cpu')
                    data = data.flatten()
                    bn[index:(index + size)] = data
                    index += size

        y, i = torch.sort(bn)  # descending=True 从大到小排序，descending=False 从小到大排序，默认 descending=Flase
        thre_index = int(total * prune_scale_rate)
        threshold = y[thre_index]  # 小于阈值部分的权重置为0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    m1 = m.weight.data.abs() > threshold
                    m.weight.grad.data *= m1
                    m.weight.data *= m1

    def sparsity_mask(self, sr):  # 稀疏化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    m.weight.grad.data.add_(sr * torch.sign(m.weight.data))


num_classes = 2

UNet_resnet18_cfg = [
    # ## -------------Encoder-------------
    # self.conv_bn_mask_relu = ConvBNReLU_mask(3, 64, kernel=3, stride=2, padding=1)
    [-1, ConvBNReLU_mask, [3, 64, 3, 2, 1]],  # 0
    # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    [-1, MaxPool2d, [2, 2]],  # 1
    # self.layer1_1 = Bottleneck_mask_RM(64, 64)
    [-1, Bottleneck_mask_RM, [64, 64]],  # 2
    # self.layer1_2 = Bottleneck_mask_RM(64, 64)
    [-1, Bottleneck_mask_RM, [64, 64]],  # 3
    # self.layer2_1 = ResBlock(64, 128, stride=2)
    [-1, ResBlock, [64, 128, 2]],  # 4
    # self.layer2_2 = Bottleneck_mask_RM(128, 128)
    [-1, Bottleneck_mask_RM, [128, 128]],  # 5
    ## -------------Decoder--------------
    # self.decoder3_conv = Conv2d(128, 64)
    [-1, Conv2d, [128, 64, 3, 1, 1]],  # 6
    # self.decoder3_upsample = ConvTranspose2d(64, 64)
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64, False]],  # 7
    # self.decoder3_cat = Concat()
    [[-1, 3], Concat, [1]],  # 8
    # self.decoder3_block_unit = ResBlock(128, 64, 1)
    [-1, ResBlock, [128, 64, 1]],  # 9
    # self.decoder4_conv = Conv2d(64, 64)
    [-1, Conv2d, [64, 64, 3, 1, 1]],  # 10
    # self.decoder4_upsample = ConvTranspose2d(64, 64)
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64, False]],  # 11
    # self.decoder4_cat = Concat()
    [[-1, 0], Concat, [1]],  # 12
    # self.decoder4_block_unit = ResBlock(128, 64, 1)
    [-1, ResBlock, [128, 64, 1]],  # 13
    # self.conv_bn_mask5 = ConvBN_mask(64, 32, kernel=3, stride=1, padding=1)
    [-1, ConvBN_mask, [64, 32, 3, 1, 1]],  # 14
    # self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
    [-1, Conv2d, [32, num_classes, 3, 1, 1]],  # 15
    # self.deconv7 = ConvTranspose2d(num_classes, num_classes)
    [-1, ConvTranspose2d, [num_classes, num_classes, 4, 2, 1, 0, num_classes, False]],  # 16

]

UNet_resnet18_RM_Prune_cfg = [
    # ## -------------Encoder-------------
    # self.conv_bn_mask_relu = ConvBNReLU_mask(3, 64, kernel=3, stride=2, padding=1)
    [-1, ConvBNReLU, [3, 64, 3, 2, 1]],  # 0-P1/2
    # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    [-1, MaxPool2d, [2, 2]],  # 4-P3/8
    # self.layer1_1 = Bottleneck_mask_RM(64, 64)
    [-1, Bottleneck_RM_prune, [64, 64]],  # 1-P2/4
    # self.layer1_2 = Bottleneck_mask_RM(64, 64)
    [-1, Bottleneck_RM_prune, [64, 64]],
    # self.layer2_1 = ResBlock(64, 128, stride=2)
    [-1, ResBlock_RM_prune, [64, 128, 2]],
    # self.layer2_2 = Bottleneck_mask_RM(128, 128)
    [-1, Bottleneck_RM_prune, [128, 128]],
    ## -------------Decoder--------------
    # self.decoder3_conv = Conv2d(128, 64)
    [-1, Conv2d, [128, 64, 3, 1, 1]],
    # self.decoder3_upsample = ConvTranspose2d(64, 64)
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64, False]],  #
    # self.decoder3_cat = Concat()
    [[-1, 3], Concat, [1]],
    # self.decoder3_block_unit = ResBlock(128, 64, 1)
    [-1, ResBlock_RM_prune, [128, 64, 1]],
    # self.decoder4_conv = Conv2d(64, 64)
    [-1, Conv2d, [64, 64, 3, 1, 1]],  # 8
    # self.decoder4_upsample = ConvTranspose2d(64, 64)
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64, False]],  #
    # self.decoder4_cat = Concat()
    [[-1, 0], Concat, [1]],
    # self.decoder4_block_unit = ResBlock(128, 64, 1)
    [-1, ResBlock_RM_prune, [128, 64, 1]],  # 9
    # self.conv_bn_mask5 = ConvBN_mask(64, 32, kernel=3, stride=1, padding=1)
    [-1, ConvBN, [64, 32, 3, 1, 1]],  # 11
    # self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
    [-1, Conv2d, [32, num_classes, 3, 1, 1]],  # 8
    # self.deconv7 = ConvTranspose2d(num_classes, num_classes)
    [-1, ConvTranspose2d, [num_classes, num_classes, 4, 2, 1, 0, num_classes, False]],  #

]

UNet_resnet18_cfg_modify = [
    # ## -------------Encoder-------------
    # self.conv_bn_mask_relu = ConvBNReLU_mask(3, 64, kernel=3, stride=2, padding=1)
    [-1, ConvBNReLU_mask, [3, 64, 3, 2, 1]],  # 0 256->128
    [-1, Conv2d, [64, 16, 1, 1, 0]],  # add:因为剪枝后链接concat精度下降严重，不知道怎么解决。所以增加一个不需要剪枝模块
    # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    [-1, MaxPool2d, [2, 2]],  # 1 128->64
    # self.layer1_1 = Bottleneck_mask_RM(64, 64)
    [-1, ResBlock, [16, 64, 1]],  # 2 64->64
    # self.layer1_2 = Bottleneck_mask_RM(64, 64)
    [-1, ResBlock, [64, 64, 1]],  # 3 64->64
    [-1, Conv2d, [64, 16, 1, 1, 0]],  # add:因为剪枝后链接concat精度下降严重，不知道怎么解决。所以增加一个不需要剪枝模块
    # self.layer2_1 = ResBlock(64, 128, stride=2)
    [-1, ResBlock, [16, 128, 2]],  # 4 64->32
    # self.layer2_2 = Bottleneck_mask_RM(128, 128)
    [-1, ResBlock, [128, 128, 1]],  # 5 32->32
    ## -------------Decoder--------------
    # self.decoder3_conv = Conv2d(128, 64)
    [-1, Conv2d, [128, 64, 3, 1, 1]],  # 6 32->32
    # self.decoder3_upsample = ConvTranspose2d(64, 64)
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64, False]],  # 7 32->64
    # self.decoder3_cat = Concat()
    [[-1, 5], Concat, [1]],  # 8 64->64
    # self.decoder3_block_unit = ResBlock(128, 64, 1)
    [-1, ResBlock, [64+16, 64, 1]],  # 9  64->64
    # self.decoder4_conv = Conv2d(64, 64)
    # [-1, Conv2d, [64, 64, 3, 1, 1]],  # 10 64->64
    [-1, Conv2d, [64, 32, 3, 1, 1]],  # 10 64->64
    # self.decoder4_upsample = ConvTranspose2d(64, 64)
    # [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64, False]],  # 11 64->128
    [-1, ConvTranspose2d, [32, 32, 4, 2, 1, 0, 32, False]],  # 11 64->128
    # self.decoder4_cat = Concat()
    [[-1, 1], Concat, [1]],  # 12 128->128
    # self.decoder4_block_unit = ResBlock(128, 64, 1)
    # [-1, ResBlock, [64+16, 64, 1]],  # 13 128->128
    [-1, ResBlock, [32+16, 48, 1]],  # 13 128->128
    # self.conv_bn_mask5 = ConvBN_mask(64, 32, kernel=3, stride=1, padding=1)
    # [-1, ConvBN_mask, [64, 32, 3, 1, 1]],  # 14 128->128
    [-1, ConvBN_mask, [48, 24, 3, 1, 1]],  # 14 128->128
    # self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
    # [-1, Conv2d, [32, num_classes, 3, 1, 1]],  # 15 128->128
    [-1, Conv2d, [24, num_classes, 3, 1, 1]],  # 15 128->128
    # self.deconv7 = ConvTranspose2d(num_classes, num_classes)
    [-1, ConvTranspose2d, [num_classes, num_classes, 4, 2, 1, 0, num_classes, False]],  # 16

]

UNet_resnet18_RM_Prune_cfg_modify = [
    # ## -------------Encoder-------------
    # self.conv_bn_mask_relu = ConvBNReLU_mask(3, 64, kernel=3, stride=2, padding=1)
    [-1, ConvBNReLU, [3, 64, 3, 2, 1]],  # 0-P1/2
    [-1, Conv2d, [64, 16, 1, 1, 0]],  # add:因为剪枝后链接concat精度下降严重，不知道怎么解决。所以增加一个不需要剪枝模块
    # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    [-1, MaxPool2d, [2, 2]],  # 4-P3/8
    # self.layer1_1 = Bottleneck_mask_RM(64, 64)
    [-1, ResBlock_RM_prune, [16, 64, 1]],  # 1-P2/4
    # self.layer1_2 = Bottleneck_mask_RM(64, 64)
    [-1, ResBlock_RM_prune, [64, 64, 1]],
    [-1, Conv2d, [64, 16, 1, 1, 0]],  # add:因为剪枝后链接concat精度下降严重，不知道怎么解决。所以增加一个不需要剪枝模块
    # self.layer2_1 = ResBlock(64, 128, stride=2)
    [-1, ResBlock_RM_prune, [16, 128, 2]],
    # self.layer2_2 = Bottleneck_mask_RM(128, 128)
    [-1, ResBlock_RM_prune, [128, 128, 1]],
    ## -------------Decoder--------------
    # self.decoder3_conv = Conv2d(128, 64)
    [-1, Conv2d, [128, 64, 3, 1, 1]],
    # self.decoder3_upsample = ConvTranspose2d(64, 64)
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64, False]],  #
    # self.decoder3_cat = Concat()
    [[-1, 5], Concat, [1]],
    # self.decoder3_block_unit = ResBlock(128, 64, 1)
    [-1, ResBlock_RM_prune, [64+16, 64, 1]],
    # self.decoder4_conv = Conv2d(64, 64)
    [-1, Conv2d, [64, 64, 3, 1, 1]],  # 8
    # self.decoder4_upsample = ConvTranspose2d(64, 64)
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64, False]],  #
    # self.decoder4_cat = Concat()
    [[-1, 1], Concat, [1]],
    # self.decoder4_block_unit = ResBlock(128, 64, 1)
    [-1, ResBlock_RM_prune, [64+16, 64, 1]],  # 9
    # self.conv_bn_mask5 = ConvBN_mask(64, 32, kernel=3, stride=1, padding=1)
    [-1, ConvBN, [64, 32, 3, 1, 1]],  # 11
    # self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
    [-1, Conv2d, [32, num_classes, 3, 1, 1]],  # 8
    # self.deconv7 = ConvTranspose2d(num_classes, num_classes)
    [-1, ConvTranspose2d, [num_classes, num_classes, 4, 2, 1, 0, num_classes, False]],  #

]

from models.backbonds.MobileNetV2 import InvertedResidual, InvertedResidualExp,InvertedResidualExp_RM, InvertedResidualRes,\
    InvertedResidualRes_RM, InvertedResidualNoRes, InvertedResidualNoRes_RM, SparseGate, InvertedResidualResConcat, InvertedResidualResConcat_RM
from models.backbonds.ShuffleNetV2 import InvertedResidual_shufflenetv2
from models.RM_Prune import rm_r, fuse_cbcb, InvertedResidual_2_3, InvertedResidual_4_5_6


UNet_MobileNetV2 = [
    # ## -------------Encoder-------------
    # self.conv_bn_mask_relu
    [-1, ConvBNReLU, [3, 32, 3, 2, 1]],  # 0 256->128
    # InvertedResidual(inp, oup, stride, expansion)
    [-1, InvertedResidual, [32, 16, 1, 1]],  # 1

    # # InvertedResidualExp(dw_inp, dw_oup, dw_stride, pw_inp, pw_oup)
    # [-1, InvertedResidualExp, [32, 16, 1, 16, 16]],

    # self.layer1
    [-1, InvertedResidual, [16, 24, 2, 4]],  # 2 128->64
    #                                             merge

    # # InvertedResidualNoRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualNoRes, [16, 64, 64, 64, 2, 64, 24]],

    [-1, InvertedResidual, [24, 24, 1, 4]],  # 3  use_res_connect

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [24, 96, 96, 96, 1, 96, 24]],

    # self.layer2
    [-1, InvertedResidual, [24, 32, 2, 4]],  # 4 64->32
    #                                             merge

    # # InvertedResidualNoRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualNoRes, [24, 96, 96, 96, 2, 96, 32]],

    [-1, InvertedResidual, [32, 32, 1, 4]],  # 5  use_res_connect
    #                                             merge

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [32, 128, 128, 128, 1, 128, 32]],

    [-1, InvertedResidual, [32, 32, 1, 4]],  # 6  use_res_connect

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [32, 128, 128, 128, 1, 128, 32]],

    # self.layer3
    [-1, InvertedResidual, [32, 64, 2, 4]],  # 7  32 -> 16 use_res_connect
    #                                             merge

    # # InvertedResidualNoRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualNoRes, [32, 128, 128, 128, 2, 128, 64]],

    [-1, InvertedResidual, [64, 64, 1, 4]],  # 8  use_res_connect

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [64, 256, 256, 256, 1, 256, 64]],

    [-1, InvertedResidual, [64, 64, 1, 4]],  # 9  use_res_connect

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [64, 256, 256, 256, 1, 256, 64]],

    [-1, InvertedResidual, [64, 64, 1, 4]],  # 10  use_res_connect

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [64, 256, 256, 256, 1, 256, 64]],

    [-1, InvertedResidual, [64, 96, 1, 4]],  # 8  use_res_connect

    # # InvertedResidualNoRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualNoRes, [64, 256, 256, 256, 1, 256, 96]],

    [-1, InvertedResidual, [96, 96, 1, 4]],  # 9  use_res_connect

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [64, 256, 256, 256, 1, 256, 64]],

    [-1, InvertedResidual, [96, 96, 1, 4]],  # 10  use_res_connect

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [96, 384, 384, 384, 1, 384, 64]],

    ## -------------Decoder--------------
    # self.decoder3_conv = Conv2d(128, 64)

    [-1, Conv2d, [96, 32, 1, 1, 0]],  # 6 32->32
    # self.decoder3_upsample
    [-1, ConvTranspose2d, [32, 32, 4, 2, 1, 0, 32, False]],  # 7 32->64
    # self.decoder3_cat = Concat()
    [[-1, 6], Concat, [1]],  # 8 64->64
    # self.decoder3_block_unit
    [-1, InvertedResidual, [64, 32, 1, 4]],  # 9  64->64

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualNoRes, [64, 256, 256, 256, 1, 256, 32]],

    [-1, Conv2d, [32, 24, 1, 1, 0]],  # 6 32->32
    # self.decoder3_upsample
    [-1, ConvTranspose2d, [24, 24, 4, 2, 1, 0, 24, False]],  # 7 32->64
    # self.decoder3_cat = Concat()
    [[-1, 3], Concat, [1]],  # 8 64->64
    # self.decoder3_block_unit
    [-1, InvertedResidual, [48, 24, 1, 4]],  # 9  64->64

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualNoRes, [48, 192, 192, 192, 1, 192, 24]],

    # self.decoder4_conv = Conv2d(64, 64)
    [-1, Conv2d, [24, 16, 1, 1, 0]],  # 10 64->64
    # self.decoder4_upsample
    [-1, ConvTranspose2d, [16, 16, 4, 2, 1, 0, 16, False]],  # 11 64->128
    # self.decoder4_cat = Concat()
    [[-1, 1], Concat, [1]],  # 12 128->128
    # self.decoder4_block_unit
    [-1, InvertedResidual, [16+16, 16, 1, 4]],  # 13 128->128

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualNoRes, [32, 128, 128, 128, 1, 128, 32]],

    # self.conv_bn_mask5 = ConvBN_mask(64, 32, kernel=3, stride=1, padding=1)
    [-1, ConvBN, [16, 3, 3, 1, 1]],  # 14 128->128
    # self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
    [-1, Conv2d, [3, num_classes, 1, 1, 0]],  # 15 128->128
    # self.deconv7 = ConvTranspose2d(num_classes, num_classes)
    [-1, ConvTranspose2d, [num_classes, num_classes, 4, 2, 1, 0, num_classes, False]],  # 16

]

UNet_MobileNetV2_prune = [
    # ## -------------Encoder-------------
    # self.conv_bn_mask_relu
    [-1, ConvBNReLU, [3, 32, 3, 2, 1]],  # 0 256->128

    # InvertedResidualExp(dw_inp, dw_oup, dw_stride, pw_inp, pw_oup)
    [-1, InvertedResidualExp, [32, 16, 1, 16, 16]],  # 1

    # self.layer1

    # InvertedResidualNoRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes, [16, 64, 64, 64, 2, 64, 24]],  # 2  128->64


    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [24, 96, 96, 96, 1, 96, 24]],  # 3
    [-1, InvertedResidualResConcat, [24, 96, 96, 96, 1, 96, 24, 24+24, 24]],  # 3

    # self.layer2

    # # InvertedResidualNoRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes, [24, 96, 96, 96, 2, 96, 32]],  # 4  64->32


    # InvertedResidualResConcat(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [32, 128, 128, 128, 1, 128, 32]],  # 5
    [-1, InvertedResidualResConcat, [32, 128, 128, 128, 1, 128, 32, 32+32, 32]],  # 5


    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [32, 128, 128, 128, 1, 128, 32]],  # 6
    [-1, InvertedResidualResConcat, [32, 128, 128, 128, 1, 128, 32, 32+32, 32]],  # 6

    # self.layer3

    # # InvertedResidualNoRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes, [32, 128, 128, 128, 2, 128, 64]],  # 7  32->16

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [64, 256, 256, 256, 1, 256, 64]],  # 8
    [-1, InvertedResidualResConcat, [64, 256, 256, 256, 1, 256, 64, 64+64, 64]],  # 8

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [64, 256, 256, 256, 1, 256, 64]],  # 9
    [-1, InvertedResidualResConcat, [64, 256, 256, 256, 1, 256, 64, 64+64, 64]],  # 9

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [64, 256, 256, 256, 1, 256, 64]],  # 10
    [-1, InvertedResidualResConcat, [64, 256, 256, 256, 1, 256, 64, 64+64, 64]],  # 10

    # self.layer4
    # # # InvertedResidualNoRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualNoRes, [64, 256, 256, 256, 1, 256, 96]],  # 11
    #
    # # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [96, 384, 384, 384, 1, 384, 96]],  # 12
    #
    # # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes, [96, 384, 384, 384, 1, 384, 96]],  # 13

    ## -------------Decoder--------------
    # self.decoder3_conv = Conv2d(128, 64)

    [-1, Conv2d, [64, 32, 1, 1, 0]],  # 11 32->32
    # self.decoder3_upsample
    [-1, ConvTranspose2d, [32, 32, 4, 2, 1, 0, 32, False]],  # 12 32->64
    # self.decoder3_cat = Concat()
    [[-1, 6], Concat, [1]],  # 13 64->64
    # self.decoder3_block_unit

    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes, [64, 256, 256, 256, 1, 256, 32]],  # 14

    [-1, Conv2d, [32, 24, 1, 1, 0]],  # 15 32->32
    # self.decoder3_upsample
    [-1, ConvTranspose2d, [24, 24, 4, 2, 1, 0, 24, False]],  # 16 32->64
    # self.decoder3_cat = Concat()
    [[-1, 3], Concat, [1]],  # 17 64->64
    # self.decoder3_block_unit
    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes, [48, 192, 192, 192, 1, 192, 24]],  # 18

    # self.decoder4_conv = Conv2d(64, 64)
    [-1, Conv2d, [24, 16, 1, 1, 0]],  # 19 64->64
    # self.decoder4_upsample
    [-1, ConvTranspose2d, [16, 16, 4, 2, 1, 0, 16, False]],  # 20 64->128
    # self.decoder4_cat = Concat()
    [[-1, 1], Concat, [1]],  # 21 128->128
    # self.decoder4_block_unit
    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes, [32, 128, 128, 128, 1, 128, 16]],  # 22

    # self.conv_bn_mask5 = ConvBN_mask(64, 32, kernel=3, stride=1, padding=1)
    [-1, ConvBN, [16, 3, 3, 1, 1]],  # 23 128->128
    # self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
    [-1, Conv2d, [3, num_classes, 1, 1, 0]],  # 24 128->128
    # self.deconv7 = ConvTranspose2d(num_classes, num_classes)
    [-1, ConvTranspose2d, [num_classes, num_classes, 4, 2, 1, 0, num_classes, False]],  # 25

]

UNet_MobileNetV2_prune_RM = [
    # ## -------------Encoder-------------
    # self.conv_bn_mask_relu
    [-1, ConvBNReLU, [3, 32, 3, 2, 1]],  # 0 256->128

    # InvertedResidualExp_RM(dw_inp, dw_oup, dw_stride, pw_inp, pw_oup)
    [-1, InvertedResidualExp_RM, [32, 16, 1, 16, 16]],  # 1

    # self.layer1

    # InvertedResidualNoRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes_RM, [16, 64, 64, 64, 2, 64, 24]],  # 2  128->64


    # InvertedResidualRes(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes_RM, [24, 96, 96, 96, 1, 96, 24]],  # 3
    [-1, InvertedResidualResConcat_RM, [24, 96, 96, 96, 1, 96, 24, 48, 24]],  # 3

    # self.layer2

    # # InvertedResidualNoRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes_RM, [24, 96, 96, 96, 2, 96, 32]],  # 4  64->32


    # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes_RM, [32, 128, 128, 128, 1, 128, 32]],  # 5
    [-1, InvertedResidualResConcat_RM, [32, 128, 128, 128, 1, 128, 32, 64, 32]],  # 5


    # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes_RM, [32, 128, 128, 128, 1, 128, 32]],  # 6
    [-1, InvertedResidualResConcat_RM, [32, 128, 128, 128, 1, 128, 32, 64, 32]],  # 6

    # self.layer3

    # # InvertedResidualNoRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes_RM, [32, 128, 128, 128, 2, 128, 64]],  # 7  32->16

    # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes_RM, [64, 256, 256, 256, 1, 256, 64]],  # 8
    [-1, InvertedResidualResConcat_RM, [64, 256, 256, 256, 1, 256, 64, 128, 64]],  # 8

    # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes_RM, [64, 256, 256, 256, 1, 256, 64]],  # 9
    [-1, InvertedResidualResConcat_RM, [64, 256, 256, 256, 1, 256, 64, 128, 64]],  # 9

    # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes_RM, [64, 256, 256, 256, 1, 256, 64]],  # 10
    [-1, InvertedResidualResConcat_RM, [64, 256, 256, 256, 1, 256, 64]],  # 10

    # self.layer4
    # # # InvertedResidualNoRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualNoRes_RM, [64, 256, 256, 256, 1, 256, 96]],  # 11
    #
    # # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes_RM, [96, 384, 384, 384, 1, 384, 96]],  # 12
    #
    # # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    # [-1, InvertedResidualRes_RM, [96, 384, 384, 384, 1, 384, 96]],  # 13

    ## -------------Decoder--------------
    # self.decoder3_conv = Conv2d(128, 64)

    [-1, Conv2d, [64, 32, 1, 1, 0]],  # 11 32->32
    # self.decoder3_upsample
    [-1, ConvTranspose2d, [32, 32, 4, 2, 1, 0, 32, False]],  # 12 32->64
    # self.decoder3_cat = Concat()
    [[-1, 6], Concat, [1]],  # 13 64->64
    # self.decoder3_block_unit

    # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes_RM, [64, 256, 256, 256, 1, 256, 32]],  # 14

    [-1, Conv2d, [32, 24, 1, 1, 0]],  # 15 32->32
    # self.decoder3_upsample
    [-1, ConvTranspose2d, [24, 24, 4, 2, 1, 0, 24, False]],  # 16 32->64
    # self.decoder3_cat = Concat()
    [[-1, 3], Concat, [1]],  # 17 64->64
    # self.decoder3_block_unit
    # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes_RM, [48, 192, 192, 192, 1, 192, 24]],  # 18

    # self.decoder4_conv = Conv2d(64, 64)
    [-1, Conv2d, [24, 16, 1, 1, 0]],  # 19 64->64
    # self.decoder4_upsample
    [-1, ConvTranspose2d, [16, 16, 4, 2, 1, 0, 16, False]],  # 20 64->128
    # self.decoder4_cat = Concat()
    [[-1, 1], Concat, [1]],  # 21 128->128
    # self.decoder4_block_unit
    # InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    [-1, InvertedResidualNoRes_RM, [32, 128, 128, 128, 1, 128, 16]],  # 22

    # self.conv_bn_mask5 = ConvBN_mask(64, 32, kernel=3, stride=1, padding=1)
    [-1, ConvBN, [16, 3, 3, 1, 1]],  # 23 128->128
    # self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
    [-1, Conv2d, [3, num_classes, 1, 1, 0]],  # 24 128->128
    # self.deconv7 = ConvTranspose2d(num_classes, num_classes)
    [-1, ConvTranspose2d, [num_classes, num_classes, 4, 2, 1, 0, num_classes, False]],  # 25

]


class MCnet_resnet18(nn.Module):
    def __init__(self, block_cfg=UNet_MobileNetV2_prune, num_classes=2):
        super(MCnet_resnet18, self).__init__()
        layers, save = [], []
        # Build model
        for i, (from_, block, args) in enumerate(block_cfg):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        self.model, self.save = nn.Sequential(*layers), sorted(save)

    def forward(self, x):
        cache = []
        det_out = None
        for i, block in enumerate(self.model):
            # print(block)
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
                                                                             block.from_]  # calculate concat detect
            x = block(x)
            det_out = x
            cache.append(x if block.index in self.save else None)
        # if isinstance(block, ConvBNReLU_mask):
        #     print(x)
        return det_out  # det

    def update_mask(self, prune_scale_rate):  # 稀疏化训练一定epoch后把小于阈值部分的权重置为0
        total = 0
        # 统计mask数量
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    size = m.weight.data.shape[0]
                    data = m.weight.data.abs().clone()
                    data = data.to('cpu')
                    data = data.flatten()
                    bn[index:(index + size)] = data
                    index += size

        y, i = torch.sort(bn)  # descending=True 从大到小排序，descending=False 从小到大排序，默认 descending=Flase
        thre_index = int(total * prune_scale_rate)
        threshold = y[thre_index]  # 小于阈值部分的权重置为0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    # mask = m.weight.data.abs() > threshold
                    # print('prune before mask:' + str(len(mask)) + '   mask>0:' + str(int(mask.sum())))
                    thr = get_mode_4_threshold(m, threshold)    # 计算新的threshold，卷积核能被4整除，且卷积核权重为0的数量不小于8
                    m1 = m.weight.data.abs() > thr
                    m.weight.grad.data *= m1
                    m.weight.data *= m1
                    # print('prune after mask:' + str(len(m1)) + '   mask>0:' + str(int(m1.sum())))

    def sparsity_BN(self, sr):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(sr*torch.sign(m.weight.data))

    def sparsity_mask(self, sr, prune_scale_rate):  # 稀疏化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    mask = m.weight.data.abs() > 0
                    # print(int(mask.sum()))
                    if int(mask.sum()) > 8:  # 当mask中卷积核权重为0的数量小于8时，不在进行稀疏化
                        m.weight.grad.data.add_(sr * torch.sign(m.weight.data))
                        if prune_scale_rate > 0:
                            self.update_mask(prune_scale_rate)

    def get_sparse_layer(self):
        sparse_modules = []
        for m in self.modules():
            if isinstance(m, SparseGate):
                sparse_modules.append(m)
        return sparse_modules

    def get_conv_layer(self):
        conv_layers = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not (m.kernel_size == (1, 1) and m.groups != 1):
                    conv_layers.append(m)
        return conv_layers

class MCnet_resnet18_RM(nn.Module):
    def __init__(self, net_params, block_cfg=UNet_MobileNetV2_prune_RM):
        super(MCnet_resnet18_RM, self).__init__()
        layers, save = [], []
        # Build model
        for i, (from_, block, args) in enumerate(block_cfg):
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
        det_out = None

        for i, block in enumerate(self.model):
            # print(block)
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
                                                                             block.from_]  # calculate concat detect
            x = block(x)
            det_out = x
            cache.append(x if block.index in self.save else None)
        # if isinstance(block, ConvBNReLU_mask):
        #     print(x)
        return det_out  # det

    def update_mask(self, prune_scale_rate):  # 稀疏化训练一定epoch后把小于阈值部分的权重置为0
        total = 0
        # 统计mask数量
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    size = m.weight.data.shape[0]
                    data = m.weight.data.abs().clone()
                    data = data.to('cpu')
                    data = data.flatten()
                    bn[index:(index + size)] = data
                    index += size

        y, i = torch.sort(bn)  # descending=True 从大到小排序，descending=False 从小到大排序，默认 descending=Flase
        thre_index = int(total * prune_scale_rate)
        threshold = y[thre_index]  # 小于阈值部分的权重置为0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    # mask = m.weight.data.abs() > threshold
                    # print('prune before mask:' + str(len(mask)) + '   mask>0:' + str(int(mask.sum())))
                    thr = get_mode_4_threshold(m, threshold)    # 计算新的threshold，卷积核能被4整除，且卷积核权重为0的数量不小于8
                    m1 = m.weight.data.abs() > thr
                    m.weight.grad.data *= m1
                    m.weight.data *= m1
                    # print('prune after mask:' + str(len(m1)) + '   mask>0:' + str(int(m1.sum())))

    def sparsity_BN(self, sr):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(sr*torch.sign(m.weight.data))

    def sparsity_mask(self, sr, prune_scale_rate):  # 稀疏化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    mask = m.weight.data.abs() > 0
                    # print(int(mask.sum()))
                    if int(mask.sum()) > 8:  # 当mask中卷积核权重为0的数量小于8时，不在进行稀疏化
                        m.weight.grad.data.add_(sr * torch.sign(m.weight.data))
                        if prune_scale_rate > 0:
                            self.update_mask(prune_scale_rate)

    def get_sparse_layer(self):
        sparse_modules = []
        for m in self.modules():
            if isinstance(m, SparseGate):
                sparse_modules.append(m)
        return sparse_modules


# block_cfg and net_params init model structure
class MCnet_resnet18_RM_Prune(nn.Module):
    def __init__(self, net_params, block_cfg=UNet_resnet18_RM_Prune_cfg_modify):
        super(MCnet_resnet18_RM_Prune, self).__init__()
        layers, save = [], []
        # Build model
        for i, (from_, block, args) in enumerate(block_cfg):
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
        det_out = None
        for i, block in enumerate(self.model):
            # if i > 6:
            #     print(x)
            # print(block)
            if block.from_ != -1:
                # print(x.shape)
                x = cache[block.from_] if isinstance(block.from_, int) else \
                    [x if j == -1 else cache[j] for j in block.from_]  # calculate concat detect
                # print(x[0].shape)
                # print(x[1].shape)
                # print(x)
            x = block(x)
            det_out = x
            cache.append(x if block.index in self.save else None)
            # if isinstance(block, ConvBNReLU):
            #     print(x)
        return det_out  # det





UNet_MobileNetV2_RM = [
    # ## -------------Encoder-------------
    # self.conv_bn_mask_relu
    [-1, ConvBNReLU, [3, 32, 3, 2, 1]],  # 0 256->128
    # InvertedResidual(inp, oup, stride, expansion)
    [-1, InvertedResidual, [32, 16, 1, 1]],  # 1
    # self.layer1
    # [-1, InvertedResidual, [16, 24, 2, 6]],  # 2 128->64
    #                                             merge
    # [-1, InvertedResidual, [24, 24, 1, 6]],  # 3  use_res_connect
    [-1, InvertedResidual_2_3, []],  # 2  use_res_connect
    # self.layer2
    # [-1, InvertedResidual, [24, 32, 2, 6]],  # 4 64->32
    #                                             merge
    # [-1, InvertedResidual, [32, 32, 1, 6]],  # 5  use_res_connect
    #                                             merge
    # [-1, InvertedResidual, [32, 32, 1, 6]],  # 6  use_res_connect
    [-1, InvertedResidual_4_5_6, []],  # 3  use_res_connect
    # self.layer3
    ## -------------Decoder--------------
    # self.decoder3_conv = Conv2d(128, 64)
    [-1, Conv2d, [32, 24, 3, 1, 1]],  # 6 32->32
    # self.decoder3_upsample
    [-1, ConvTranspose2d, [24, 24, 4, 2, 1, 0, 24, False]],  # 7 32->64
    # self.decoder3_cat = Concat()
    [[-1, 2], Concat, [1]],  # 8 64->64
    # self.decoder3_block_unit
    [-1, InvertedResidual, [48, 24, 1, 6]],  # 9  64->64
    # self.decoder4_conv = Conv2d(64, 64)
    [-1, Conv2d, [24, 16, 3, 1, 1]],  # 10 64->64
    # self.decoder4_upsample
    [-1, ConvTranspose2d, [16, 16, 4, 2, 1, 0, 16, False]],  # 11 64->128
    # self.decoder4_cat = Concat()
    [[-1, 1], Concat, [1]],  # 12 128->128
    # self.decoder4_block_unit
    [-1, InvertedResidual, [16+16, 16, 1, 6]],  # 13 128->128
    # self.conv_bn_mask5 = ConvBN_mask(64, 32, kernel=3, stride=1, padding=1)
    [-1, ConvBN, [16, 8, 3, 1, 1]],  # 14 128->128
    # self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
    [-1, Conv2d, [8, num_classes, 3, 1, 1]],  # 15 128->128
    # self.deconv7 = ConvTranspose2d(num_classes, num_classes)
    [-1, ConvTranspose2d, [num_classes, num_classes, 4, 2, 1, 0, num_classes, False]],  # 16

]


def merge_conv_bn(features):
    new_features = []
    while features:
        if len(features) > 3 and isinstance(features[0], nn.Conv2d) and isinstance(features[1], nn.BatchNorm2d) and \
                isinstance(features[2], nn.Conv2d) and isinstance(features[3], nn.BatchNorm2d) and \
                features[0].out_channels == features[2].in_channels and features[2].out_channels > features[2].in_channels:
            print(str(features[0].out_channels)+': '+str(features[2].in_channels))
            conv, bn = fuse_cbcb(features[0], features[1], features[2], features[3])
            print(str(conv.in_channels) + ': ' + str(conv.out_channels))
            new_features.append(conv)
            new_features.append(bn)
            features = features[4:]
        else:
            new_features.append(features.pop(0))
    return new_features

def mobilenetv2_to_mobilenetv1(model):
    index = 0
    new_model = MCnet_resnet18(block_cfg=UNet_MobileNetV2_RM)
    for i, m in enumerate(model.model):
        if i in [2, 3, 4, 5, 6]:
            features = []
            if i == 2:
                for mm in m.modules():
                    if not list(mm.children()):
                        features.append(mm)

                features += rm_r(model.model[i + 1])
                new_features = merge_conv_bn(copy.deepcopy(features))
                print(new_features)
                # conv1+bn+relu
                new_model.model[index].block[0].weight.data = new_features[0].weight.data
                new_model.model[index].block[1].weight.data = new_features[1].weight.data
                new_model.model[index].block[1].bias.data = new_features[1].bias.data
                new_model.model[index].block[1].running_mean = new_features[1].running_mean
                new_model.model[index].block[1].running_var = new_features[1].running_var
                # conv1+bn+relu
                new_model.model[index].block[3].weight.data = new_features[3].weight.data
                new_model.model[index].block[4].weight.data = new_features[4].weight.data
                new_model.model[index].block[4].bias.data = new_features[4].bias.data
                new_model.model[index].block[4].running_mean = new_features[4].running_mean
                new_model.model[index].block[4].running_var = new_features[4].running_var

                # conv1+bn+prelu
                new_model.model[index].block[6].weight.data = new_features[6].weight.data
                new_model.model[index].block[7].weight.data = new_features[7].weight.data
                new_model.model[index].block[7].bias.data = new_features[7].bias.data
                new_model.model[index].block[7].running_mean = new_features[7].running_mean
                new_model.model[index].block[7].running_var = new_features[7].running_var
                new_model.model[index].block[8].weight.data = new_features[8].weight.data

                # conv1+bn+prelu
                new_model.model[index].block[9].weight.data = new_features[9].weight.data
                new_model.model[index].block[10].weight.data = new_features[10].weight.data
                new_model.model[index].block[10].bias.data = new_features[10].bias.data
                new_model.model[index].block[10].running_mean = new_features[10].running_mean
                new_model.model[index].block[10].running_var = new_features[10].running_var
                new_model.model[index].block[11].weight.data = new_features[11].weight.data

                # conv1+bn
                new_model.model[index].block[12].weight.data = new_features[12].weight.data
                new_model.model[index].block[13].weight.data = new_features[13].weight.data
                new_model.model[index].block[13].bias.data = new_features[13].bias.data
                new_model.model[index].block[13].running_mean = new_features[13].running_mean
                new_model.model[index].block[13].running_var = new_features[13].running_var
                index += 1
            if i == 4:
                for mm in m.modules():
                    if not list(mm.children()):
                        features.append(mm)

                features += rm_r(model.model[i + 1])
                features += rm_r(model.model[i + 2])
                new_features = merge_conv_bn(copy.deepcopy(features))
                print(new_features)

                # conv1+bn+relu
                new_model.model[index].block[0].weight.data = new_features[0].weight.data
                new_model.model[index].block[1].weight.data = new_features[1].weight.data
                new_model.model[index].block[1].bias.data = new_features[1].bias.data
                new_model.model[index].block[1].running_mean = new_features[1].running_mean
                new_model.model[index].block[1].running_var = new_features[1].running_var
                # conv1+bn+relu
                new_model.model[index].block[3].weight.data = new_features[3].weight.data
                new_model.model[index].block[4].weight.data = new_features[4].weight.data
                new_model.model[index].block[4].bias.data = new_features[4].bias.data
                new_model.model[index].block[4].running_mean = new_features[4].running_mean
                new_model.model[index].block[4].running_var = new_features[4].running_var
                # conv1+bn+prelu
                new_model.model[index].block[6].weight.data = new_features[6].weight.data
                new_model.model[index].block[7].weight.data = new_features[7].weight.data
                new_model.model[index].block[7].bias.data = new_features[7].bias.data
                new_model.model[index].block[7].running_mean = new_features[7].running_mean
                new_model.model[index].block[7].running_var = new_features[7].running_var
                new_model.model[index].block[8].weight.data = new_features[8].weight.data

                # conv1+bn+prelu
                new_model.model[index].block[9].weight.data = new_features[9].weight.data
                new_model.model[index].block[10].weight.data = new_features[10].weight.data
                new_model.model[index].block[10].bias.data = new_features[10].bias.data
                new_model.model[index].block[10].running_mean = new_features[10].running_mean
                new_model.model[index].block[10].running_var = new_features[10].running_var
                new_model.model[index].block[11].weight.data = new_features[11].weight.data

                # conv1+bn+prelu
                new_model.model[index].block[12].weight.data = new_features[12].weight.data
                new_model.model[index].block[13].weight.data = new_features[13].weight.data
                new_model.model[index].block[13].bias.data = new_features[13].bias.data
                new_model.model[index].block[13].running_mean = new_features[13].running_mean
                new_model.model[index].block[13].running_var = new_features[13].running_var
                new_model.model[index].block[14].weight.data = new_features[14].weight.data

                # conv1+bn+prelu
                new_model.model[index].block[15].weight.data = new_features[15].weight.data
                new_model.model[index].block[16].weight.data = new_features[16].weight.data
                new_model.model[index].block[16].bias.data = new_features[16].bias.data
                new_model.model[index].block[16].running_mean = new_features[16].running_mean
                new_model.model[index].block[16].running_var = new_features[16].running_var
                new_model.model[index].block[17].weight.data = new_features[17].weight.data

                # conv1+bn
                new_model.model[index].block[18].weight.data = new_features[18].weight.data
                new_model.model[index].block[19].weight.data = new_features[19].weight.data
                new_model.model[index].block[19].bias.data = new_features[19].bias.data
                new_model.model[index].block[19].running_mean = new_features[19].running_mean
                new_model.model[index].block[19].running_var = new_features[19].running_var
                index += 1

        else:
            if isinstance(m, Concat):
                print()
            else:
                new_model.model[index] = m
            index += 1

    return new_model


UNet_shuffleNetV2 = [
    # ## -------------Encoder-------------
    # self.conv_bn_mask_relu
    [-1, ConvBNReLU, [3, 24, 3, 2, 1]],  # 0 256->128
    # InvertedResidual_shufflenetv2(inp, oup, stride)
    [-1, InvertedResidual_shufflenetv2, [24, 24, 1]],  # 1
    # self.layer1
    [-1, InvertedResidual_shufflenetv2, [24, 48, 2]],  # 2 128->64
    #                                             merge
    [-1, InvertedResidual_shufflenetv2, [48, 48, 1]],  # 3  use_res_connect
    # self.layer2
    [-1, InvertedResidual_shufflenetv2, [48, 96, 2]],  # 4 64->32
    #                                             merge
    [-1, InvertedResidual_shufflenetv2, [96, 96, 1]],  # 5  use_res_connect
    #                                             merge
    [-1, InvertedResidual_shufflenetv2, [96, 96, 1]],  # 6  use_res_connect
    # self.layer3
    ## -------------Decoder--------------
    # self.decoder3_conv = Conv2d(128, 64)
    [-1, Conv2d, [96, 24, 3, 1, 1]],  # 6 32->32
    # self.decoder3_upsample
    [-1, ConvTranspose2d, [24, 24, 4, 2, 1, 0, 24, False]],  # 7 32->64
    # self.decoder3_cat = Concat()
    [[-1, 3], Concat, [1]],  # 8 64->64
    # self.decoder3_block_unit
    [-1, InvertedResidual_shufflenetv2, [24+48, 72, 1]],  # 9  64->64
    # self.decoder4_conv = Conv2d(64, 64)
    [-1, Conv2d, [72, 16, 3, 1, 1]],  # 10 64->64
    # self.decoder4_upsample
    [-1, ConvTranspose2d, [16, 16, 4, 2, 1, 0, 16, False]],  # 11 64->128
    # self.decoder4_cat = Concat()
    [[-1, 1], Concat, [1]],  # 12 128->128
    # self.decoder4_block_unit
    [-1, InvertedResidual_shufflenetv2, [16+24, 40, 1]],  # 13 128->128
    # self.conv_bn_mask5 = ConvBN_mask(64, 32, kernel=3, stride=1, padding=1)
    [-1, ConvBN, [40, 8, 3, 1, 1]],  # 14 128->128
    # self.conv6 = Conv2d(32, num_classes, kernel_size=3, padding=1)
    [-1, Conv2d, [8, num_classes, 3, 1, 1]],  # 15 128->128
    # self.deconv7 = ConvTranspose2d(num_classes, num_classes)
    [-1, ConvTranspose2d, [num_classes, num_classes, 4, 2, 1, 0, num_classes, False]],  # 16

]

if __name__ == "__main__":
    model = MCnet_resnet18()
    sparse_list = model.get_sparse_layer()
    sparse_layer_weight_concat = torch.cat(list(map(lambda m: m._conv.weight.view(-1), sparse_list)))
    print(model)
    # model = UNet(backbone="mobilenetv2", num_classes=2, pretrained_backbone=None)
    # print(model)
    from torchsummary import summary
    #
    # # model = mobilenetv2_to_mobilenetv1(model)
    #
    model = model.to("cuda")
    summary(model, (3, 256, 256))
    # onnx_path = '/media/z590/G/Pytorch/Segmentation-PyTorch-master/workspace/mobilenetv2_RM.onnx'
    # model = model.to('cpu')
    # pth2onnx(model, onnx_path, 256, 256)




# 	(3, 320, 320)
# 	(backbone): mobilenetv2
# Total params: 4,683,331
# Trainable params: 4,683,331
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 1.17
# Forward/backward pass size (MB): 696.41
# Params size (MB): 17.87
# Estimated Total Size (MB): 715.45

# mobilenetv2 下采样3次
# 	(3, 256, 256)
# Total params: 94,157
# Trainable params: 94,157
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 357.69
# Params size (MB): 0.36
# Estimated Total Size (MB): 358.80

# mobilenetv2 下采样5次
# Total params: 2,779,661
# Trainable params: 2,779,661
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 447.50
# Params size (MB): 10.60
# Estimated Total Size (MB): 458.85


# resnet18 下采样3次
# 	(3, 256, 256)
# Total params: 937,821
# Trainable params: 937,821
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 184.12
# Params size (MB): 3.58
# Estimated Total Size (MB): 188.45


# mobilenetv2_RM 下采样3次
# 	(3, 256, 256)
# Total params: 168,418
# Trainable params: 168,418
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 376.75
# Params size (MB): 0.64
# Estimated Total Size (MB): 378.14


# ================================================================
# Total params: 94,810
# Trainable params: 94,810
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 368.00
# Params size (MB): 0.36
# Estimated Total Size (MB): 369.11
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# # 三层Unet——mobile
# Total params: 217,238
# Trainable params: 217,238
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 286.38
# Params size (MB): 0.83
# Estimated Total Size (MB): 287.95
# ----------------------------------------------------------------


# 三层加入了mask后
# ================================================================
# Total params: 219,710
# Trainable params: 219,710
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 336.62
# Params size (MB): 0.84
# Estimated Total Size (MB): 338.21
# ----------------------------------------------------------------