import torch
import torch.nn as nn


JacintoNetV2 = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512]


def ConvBNReLU(in_planes, out_planes, kernel_size=1, stride=1, padding=None, groups=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


def JacintoNetV2_lite(pretrained=False):
    layers = []
    in_channels = 3
    # from, module, in, out, kernel, stride, pad, group
    # [-1, 'conv', 3, 32, 5, 2, 2, 1],  # 0
    layers += [ConvBNReLU(in_channels, 24, kernel_size=5, stride=2, padding=2, groups=1)]   # 512 # 320
    # [-1, 'conv', 32, 32, 3, 1, 1, 4],  # 1
    layers += [ConvBNReLU(24, 24, kernel_size=3, stride=1, padding=1, groups=4)]
    # [-1, 'pool', 32, 32, 2, 2, 0, 1],  # 2
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]   # 256 # 160
    # [-1, 'conv', 32, 64, 3, 1, 1, 1],  # 3
    layers += [ConvBNReLU(24, 40, 3, 1, 1, 1)]
    # [-1, 'conv', 64, 64, 3, 1, 1, 4],  # 4
    layers += [ConvBNReLU(40, 40, 3, 1, 1, 4)]
    # [-1, 'pool', 64, 64, 2, 2, 0, 1],  # 5
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]   # 128 # 80
    # [-1, 'conv', 64, 128, 3, 1, 1, 1],  # 6
    layers += [ConvBNReLU(40, 80, 3, 1, 1, 1)]
    # [-1, 'conv', 128, 128, 3, 1, 1, 4],  # 7
    layers += [ConvBNReLU(80, 80, 3, 1, 1, 4)]     # 7
    # [-1, 'pool', 128, 128, 2, 2, 0, 1],  # 8
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]   # 64 # 40
    # [-1, 'conv', 128, 256, 3, 1, 1, 1],  # 9
    layers += [ConvBNReLU(80, 160, 3, 1, 1, 1)]
    # [-1, 'conv', 256, 256, 3, 1, 1, 4],  # 10
    layers += [ConvBNReLU(160, 160, 3, 1, 1, 4)]    # 10 # branch 1
    # [-1, 'pool', 256, 256, 2, 2, 0, 1],  # 11
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]   # 32
    # [-1, 'conv', 256, 512, 3, 1, 1, 1],  # 12
    layers += [ConvBNReLU(160, 320, 3, 1, 1, 1)]
    # [-1, 'conv', 512, 512, 3, 1, 1, 4]]  # 13
    layers += [ConvBNReLU(320, 320, 3, 1, 1, 4)]      # 13    # branch 2

    # pool6
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 14    # branch 3   # 16
    # pool7
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 15    # branch 4   # 8
    # pool8
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 16    # branch 5   # 4

    model = nn.ModuleList(layers)
    return model


def JacintoNetV2_yihang(pretrained=False):
    layers = []
    in_channels = 3
    # from, module, in, out, kernel, stride, pad, group
    # [-1, 'conv', 3, 32, 5, 2, 2, 1],  # 0
    layers += [ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=1, groups=1)]   # 256->128TIDL_ConvolutionLayer[numInChannels:3,numOutChannels:32, kernelW:3, kernelH:3, numGroups:1]
    # [-1, 'conv', 32, 32, 3, 1, 1, 4],  # 1
    layers += [ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1, groups=4)]#TIDL_ConvolutionLayer[numInChannels:32,numOutChannels:32,kernelW:3,kernelH:3,numGroups:4]
    # [-1, 'pool', 32, 32, 2, 2, 0, 1],  # 2
    layers += [ConvBNReLU(32, 64, 3, 2, 1, 1)]   # 128->64#TIDL_ConvolutionLayer[numInChannels:32,numOutChannels:64,kernelW:3,kernelH:3,numGroups:1]
    # [-1, 'conv', 32, 64, 3, 1, 1, 1],  # 3
    layers += [ConvBNReLU(64, 64, 3, 1, 1, 1)]  # TIDL_ConvolutionLayer[numInChannels:64,numOutChannels:64,kernelW:3,kernelH:3,numGroups:1]
    # [-1, 'conv', 64, 64, 3, 1, 1, 4],  # 4
    layers += [ConvBNReLU(64, 64, 3, 1, 1, 4)]  # TIDL_ConvolutionLayer[numInChannels:64,numOutChannels:64,kernelW:3,kernelH:3,numGroups:4]
    # [-1, 'pool', 64, 64, 2, 2, 0, 1],  # 5
    layers += [ConvBNReLU(64, 128, 3, 2, 1, 1)]   # 64->32  # TIDL_ConvolutionLayer[numInChannels:64,numOutChannels:128,kernelW:3,kernelH:3,numGroups:1]
    # [-1, 'conv', 64, 128, 3, 1, 1, 1],  # 6
    layers += [ConvBNReLU(128, 64, 3, 1, 1, 1)] # TIDL_ConvolutionLayer[numInChannels:128,numOutChannels:64,kernelW:3,kernelH:3,numGroups:1]
    # [-1, 'conv', 128, 128, 3, 1, 1, 4],  # 7
    layers += [ConvBNReLU(64, 64, 3, 1, 1, 4)]     # 7  #TIDL_ConvolutionLayer[numInChannels:64,numOutChannels:64,kernelW:3,kernelH:3,numGroups:4]
    # [-1, 'pool', 128, 128, 2, 2, 0, 1],  # 8
    layers += [ConvBNReLU(64, 256, 3, 2, 1, 1)]   # 32->16  #  TIDL_ConvolutionLayer[numInChannels:64,numOutChannels:256,kernelW:3,kernelH:3,numGroups:1]
    # [-1, 'conv', 128, 256, 3, 1, 1, 1],  # 9
    layers += [ConvBNReLU(256, 128, 3, 1, 1, 1)]    # TIDL_ConvolutionLayer[numInChannels:256,numOutChannels:128,kernelW:3,kernelH:3,numGroups:1]
    # [-1, 'conv', 256, 256, 3, 1, 1, 4],  # 10
    layers += [ConvBNReLU(128, 128, 3, 1, 1, 4)]    # 10 # branch 1 # TIDL_ConvolutionLayer[numInChannels:128,numOutChannels:128,kernelW:3,kernelH:3,numGroups:4]

    model = nn.ModuleList(layers)
    return model


def JacintoNetV2_nano(pretrained=False):
    layers = []
    in_channels = 3
    # from, module, in, out, kernel, stride, pad, group
    # [-1, 'conv', 3, 32, 5, 2, 2, 1],  # 0
    layers += [ConvBNReLU(in_channels, 32, kernel_size=5, stride=2, padding=2, groups=1)]   # 256->128
    # [-1, 'conv', 32, 32, 3, 1, 1, 4],  # 1
    layers += [ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1, groups=4)]
    # [-1, 'pool', 32, 32, 2, 2, 0, 1],  # 2
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]   # 128->64
    # [-1, 'conv', 32, 64, 3, 1, 1, 1],  # 3
    layers += [ConvBNReLU(32, 64, 3, 1, 1, 1)]
    # [-1, 'conv', 64, 64, 3, 1, 1, 4],  # 4
    layers += [ConvBNReLU(64, 64, 3, 1, 1, 4)]
    # [-1, 'pool', 64, 64, 2, 2, 0, 1],  # 5
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]   # 64->32
    # [-1, 'conv', 64, 128, 3, 1, 1, 1],  # 6
    layers += [ConvBNReLU(64, 128, 3, 1, 1, 1)]
    # [-1, 'conv', 128, 128, 3, 1, 1, 4],  # 7
    layers += [ConvBNReLU(128, 128, 3, 1, 1, 4)]     # 7
    # [-1, 'pool', 128, 128, 2, 2, 0, 1],  # 8
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]   # 32->16
    # [-1, 'conv', 128, 256, 3, 1, 1, 1],  # 9
    layers += [ConvBNReLU(128, 256, 3, 1, 1, 1)]
    # [-1, 'conv', 256, 256, 3, 1, 1, 4],  # 10
    layers += [ConvBNReLU(256, 256, 3, 1, 1, 4)]    # 10 # branch 1

    model = nn.ModuleList(layers)
    return model