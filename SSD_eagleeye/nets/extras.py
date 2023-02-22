import torch
import torch.nn as nn
import torch.nn.init as init
from nets.mobilenetv2 import InvertedResidual, InvertedResidual_Quantization_Friendly
from nets.mobilenetv1 import conv_dw
from nets.JacintoNetV2 import ConvBNReLU


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def add_extras(in_channels, backbone_name):
    layers = []
    if backbone_name == 'vgg':
        # Block 6
        # 19,19,1024 -> 19,19,256 -> 10,10,512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]
        # Block 7
        # 10,10,512 -> 10,10,128 -> 5,5,256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
        # Block 8
        # 5,5,256 -> 5,5,128 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        # Block 9
        # 3,3,256 -> 3,3,128 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    elif backbone_name == 'mobilenetv1_half':
        layers += [conv_dw(128, 256, 1)]
        layers += [conv_dw(256, 256, 1)]
        layers += [conv_dw(256, 256, 1)]
        layers += [conv_dw(256, 256, 1)]
        layers += [conv_dw(256, 256, 1)]
    elif backbone_name == 'mobilenetv1':
        layers += [conv_dw(256, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
    elif backbone_name == 'xception':
        layers += [conv_dw(128, 512, 1)]
        layers += [conv_dw(256, 512, 1)]
        layers += [conv_dw(256, 512, 1)]
        layers += [conv_dw(256, 512, 1)]
        layers += [conv_dw(256, 512, 1)]
    elif backbone_name == 'mobilenetv1_MFR':
        layers += [conv_dw(256+256, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
    elif backbone_name == 'mobilenetv1_F_SSD':
        layers += [conv_dw(256+256+256, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
    elif backbone_name == 'mobilenetv2':
        layers += [InvertedResidual(in_channels, 512, stride=2, expand_ratio=0.2)]
        layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.25)]
        layers += [InvertedResidual(256, 256, stride=2, expand_ratio=0.5)]
        layers += [InvertedResidual(256, 64, stride=2, expand_ratio=0.25)]
    elif backbone_name == 'mobilenetv2_half':
        layers += [InvertedResidual(in_channels, 512, stride=2, expand_ratio=1)]
        # layers += [InvertedResidual(320, 256, stride=2, expand_ratio=0.25)]
        layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.5)]
        layers += [InvertedResidual(256, 128, stride=2, expand_ratio=0.5)]
    elif backbone_name == 'mobilenetv2_quarter':
        layers += [InvertedResidual(in_channels, 256, stride=2, expand_ratio=0.2)]
        # layers += [InvertedResidual(320, 256, stride=2, expand_ratio=0.25)]
        layers += [InvertedResidual(256, 128, stride=2, expand_ratio=0.5)]
        layers += [InvertedResidual(128, 64, stride=2, expand_ratio=0.25)]
    elif backbone_name == 'JacintoNetV2_lite':
        # branch Block 1:
        layers += [ConvBNReLU(160, 160, kernel_size=1, stride=1, padding=0)]
        # branch Block 2:
        layers += [ConvBNReLU(320, 160, kernel_size=1, stride=1, padding=0)]
        # branch Block 3:
        layers += [ConvBNReLU(320, 160, kernel_size=1, stride=1, padding=0)]
        # branch Block 4:
        layers += [ConvBNReLU(320, 160, kernel_size=1, stride=1, padding=0)]
        # branch Block 5:
        layers += [ConvBNReLU(320, 160, kernel_size=1, stride=1, padding=0)]
    elif backbone_name == 'JacintoNetV2_yihang':
        # branch Block 1:
        layers += [ConvBNReLU(64, 256, kernel_size=1, stride=1, padding=0)]     # TIDL_ConvolutionLayer[numInChannels:64,numOutChannels:256,kernelW:1,kernelH:1,numGroups:1]
        # branch Block 2:
        layers += [ConvBNReLU(128, 256, kernel_size=1, stride=1, padding=0)]    # TIDL_ConvolutionLayer[numInChannels:128,numOutChannels:256,kernelW:1,kernelH:1,numGroups:1]
    elif backbone_name == 'JacintoNetV2_nano':
        # branch Block 1:
        layers += [ConvBNReLU(128, 256, kernel_size=1, stride=1, padding=0)]
        # branch Block 2:
        layers += [ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0)]
    elif backbone_name == 'corner_point_mobilenetv2_4_tiny':
        # branch Block 1:
        layers += [InvertedResidual_Quantization_Friendly(48, 96, 1, 2)]   # sum:128->prune:115# sum:256->prune:221
        # branch Block 2:
        layers += [InvertedResidual_Quantization_Friendly(56, 128, 1, 2)]   # sum:128->prune:34# sum:256->prune:176
    elif backbone_name == 'corner_point_mobilenetv2_4':
        # branch Block 1:
        layers += [InvertedResidual_Quantization_Friendly(64, 256, 1, 2)]
        # branch Block 2:
        layers += [InvertedResidual_Quantization_Friendly(64, 256, 1, 2)]
    elif backbone_name == 'corner_point_mobilenetv2_4_advance':
        # branch Block 1:
        layers += [InvertedResidual_Quantization_Friendly(64, 256, 1, 2)]
        # branch Block 2:
        layers += [InvertedResidual_Quantization_Friendly(64, 256, 1, 2)]
    return nn.ModuleList(layers)




# MFRDet
class MFR(nn.Module):
    def __init__(self, inp):
        super(MFR, self).__init__()
        oup = int(0.5*inp)
        self.conv = ConvBNReLU(inp, oup, kernel_size=1, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(oup, oup, kernel_size=4, padding=1, stride=2, groups=oup)  # bias=True

    def forward(self, x0, x1):
        x1 = self.conv(x1)
        x1 = self.deconv(x1)
        # print(x0.shape)
        # print(x1.shape)
        out = torch.cat((x0, x1), 1)
        return out


class F_SSD(nn.Module):
    def __init__(self, inp0, inp1):    # 128,512
        super(F_SSD, self).__init__()
        oup0 = int(2*inp0)
        self.down_conv0 = ConvBNReLU(inp0, inp0, kernel_size=3, stride=2, padding=1, groups=inp0)
        self.down_conv1 = ConvBNReLU(inp0, oup0, kernel_size=1, stride=1, padding=0)

        oup1 = int(0.5 * inp1)
        self.up_conv1 = ConvBNReLU(inp1, oup1, kernel_size=1, stride=1, padding=0)
        self.up_deconv1 = nn.ConvTranspose2d(oup1, oup1, kernel_size=4, padding=1, stride=2, groups=oup1)  # bias=True

        # self.conv = ConvBNReLU(256*3, 256*2, kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1, x2):
        x0 = self.down_conv0(x0)
        x0 = self.down_conv1(x0)

        x2 = self.up_conv1(x2)
        x2 = self.up_deconv1(x2)
        # print(x0.shape)
        # print(x1.shape)
        out = torch.cat((x0, x1, x2), 1)
        # out = self.conv(out)
        return out
