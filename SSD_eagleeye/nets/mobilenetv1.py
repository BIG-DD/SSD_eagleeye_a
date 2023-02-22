# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class conv_bn(nn.Sequential):
    def __init__(self, inp, oup, stride):
        super(conv_bn, self).__init__(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
        )
        self.out_channels = oup

class conv_dw(nn.Sequential):
    def __init__(self, inp, oup, stride):
        super(conv_dw, self).__init__(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.out_channels = oup


class conv_pool(nn.Sequential):
    def __init__(self, inp, oup, stride):
        super(conv_pool, self).__init__(
            nn.MaxPool2d(2, 2),
            #
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.out_channels = oup

class MobileNetV1_half(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1_half, self).__init__()
        self.features = nn.Sequential(
            conv_bn(3, 16, 2),  # 512->256
            conv_dw(16, 32, 1),
            conv_pool(32, 48, 2),  # 256->128
            conv_dw(48, 48, 1),
            conv_pool(48, 64, 2),  # 128->64
            conv_dw(64, 64, 1),
            conv_pool(64, 128, 2),  # 64->32
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),   # 11
            conv_pool(128, 256, 2),  # 32->16
            conv_dw(256, 256, 1),   # 13
            nn.MaxPool2d(2, 2),     # 14  # 16->8
            nn.MaxPool2d(2, 2),     # 15  # 8->4
            nn.MaxPool2d(2, 2),     # 16  # 4->2
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def mobilenet_v1_half(pretrained=False, compress=1, **kwargs):
    model = MobileNetV1_half(512)
    return model


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()
        self.features = nn.Sequential(
            conv_bn(3, 16, 2),      # 512->256
            conv_dw(16, 32, 1),
            conv_dw(32, 64, 2),     # 256->128
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),    # 128->64
            conv_dw(128, 128, 1),   # 5
            conv_dw(128, 256, 2),   # 64->32
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),   # 11
            conv_dw(256, 512, 2),   # 32->16
            conv_dw(512, 512, 1),   # 13
            nn.MaxPool2d(2, 2),     # 14  # 16->8
            nn.MaxPool2d(2, 2),     # 15  # 8->4
            nn.MaxPool2d(2, 2),     # 16  # 4->2
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def mobilenet_v1(pretrained=False, compress=1, **kwargs):
    model = MobileNetV1(512)
    return model
# ----------------------------------------------------------------
# Input size (MB): 3.00
# Forward/backward pass size (MB): 357.71
# Params size (MB): 12.08
# Estimated Total Size (MB): 372.79
# ----------------------------------------------------------------