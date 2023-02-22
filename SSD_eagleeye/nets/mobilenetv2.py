from torch import nn
import torch
from torch.ao.quantization import DeQuantStub, QuantStub
from typing import Any, List, Optional, Union

def fuse_modules(
    model: nn.Module, modules_to_fuse: Union[List[str], List[List[str]]], is_qat: Optional[bool], **kwargs: Any
):
    if is_qat is None:
        is_qat = model.training
    method = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
    return method(model, modules_to_fuse, **kwargs)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_planes


class Conv(nn.Module):
    def __init__(self, inp, oup, kernel=3, stride=1, padding=0, groups=1, quantization=False):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.quantization = quantization
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, stride, padding, groups=groups, bias=False)
        )
        if self.quantization:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.quantization:
            x = self.quant(x)
            x = self.conv(x)
            x = self.dequant(x)
            return x
        else:
            # print(self.conv[0](x))
            # print(self.conv[:2](x))
            # print(self.conv[:](x))
            return self.conv(x)
    # 算子折叠
    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                fuse_modules(self.conv, [str(idx), str(idx + 1), str(idx + 2)], inplace=True)


class ConvBNReLU1(nn.Module):
    def __init__(self, inp, oup, kernel=3, stride=1, padding=0, groups=1, quantization=False):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvBNReLU1, self).__init__()
        self.quantization = quantization
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )
        if self.quantization:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.quantization:
            x = self.quant(x)
            x = self.conv(x)
            x = self.dequant(x)
            return x
        else:
            # print(self.conv[0](x))
            # print(self.conv[:2](x))
            # print(self.conv[:](x))
            return self.conv(x)
    # 算子折叠
    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                fuse_modules(self.conv, [str(idx), str(idx + 1), str(idx + 2)], inplace=True)



class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x, y):
        return torch.cat([x, y], self.d)


class Concat1(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1, Quantization=False):
        super(Concat1, self).__init__()
        self.d = dimension

    def forward(self, x):
        """ print("***********************")
        for f in x:
            print(f.shape) """
        return torch.cat(x, self.d)

    def fuse_model(self):
        print()
# 在基于 ARM
# Concat 操作确实是一种 0 Params，0 FLOPs 的操作。但是，它在硬件设备上的计算成本是不可忽略的。
# 在硬件设备上，由于复杂的内存复制，Concat 操作比加法(Add)操作效率低得多
class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, Quantization=False):
        super(Add, self).__init__()

    def forward(self, x, y):
        """ print("***********************")
        for f in x:
            print(f.shape) """
        return torch.add(x, y)

    def fuse_model(self):
        print()


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, quantization=False):
        super(ConvTranspose2d, self).__init__()
        self.quantization = quantization
                                             # 24, 24, 4, 2, 1, 0, 24
        #                                       3,  1,  3, 2, 1, 1
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups)    # bias=True
        if self.quantization:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.quantization:
            x = self.quant(x)
            x = self.deconv(x)
            x = self.dequant(x)
            return x
        else:
            return self.deconv(x)

    def fuse_model(self):
        print()


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual_Quantization_Friendly(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, quantization=False):
        super(InvertedResidual_Quantization_Friendly, self).__init__()
        self.stride = stride
        self.quantization = quantization
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        self.out_channels = oup
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        if not self.use_res_connect:
            self.relu = nn.ReLU(inplace=True)
        if self.quantization:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        # print('InvertedResidual')
        if self.quantization:
            x = self.quant(x)
            if self.use_res_connect:
                x = self.skip_add.add(x, self.conv(x))
            else:
                x = self.relu(self.conv(x))
            x = self.dequant(x)
            return x
        else:
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.relu(self.conv(x))

    # 算子折叠
    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                if idx+2 < len(self.conv):
                    fuse_modules(self.conv, [str(idx), str(idx + 1), str(idx + 2)], inplace=True)
                else:
                    fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)


class InvertedResidual_prune(nn.Module):
    def __init__(self, layer_params, expansion, use_res_connect):
        super(InvertedResidual_prune, self).__init__()
        self.use_res_connect = use_res_connect
        if expansion == 1:
            self.conv = nn.Sequential(
				# dw
				nn.Conv2d(layer_params[0][0], layer_params[0][1], 3, layer_params[0][3], 1, groups=layer_params[0][5], dilation=1, bias=False),
				# nn.BatchNorm2d(layer_params[0][1], ),
				# nn.ReLU(inplace=True),
				# pw-linear
				nn.Conv2d(layer_params[1][0], layer_params[1][1], 1, 1, 0, bias=False),
				nn.BatchNorm2d(layer_params[1][1]),
			)
        else:
            self.conv = nn.Sequential(
				# pw
				nn.Conv2d(layer_params[0][0], layer_params[0][1], 1, 1, 0, bias=False),
				nn.BatchNorm2d(layer_params[0][1]),
				nn.ReLU(inplace=True),
				# dw
				nn.Conv2d(layer_params[1][0], layer_params[1][1], 3, layer_params[1][3], 1, groups=layer_params[1][5], dilation=1, bias=False),
				# nn.BatchNorm2d(layer_params[1][1]),
				# nn.ReLU(inplace=True),
				# pw-linear
				nn.Conv2d(layer_params[2][0], layer_params[2][1], 1, 1, 0, bias=False),
				nn.BatchNorm2d(layer_params[2][1]),
			)
        if not self.use_res_connect:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # print('InvertedResidual_prune')
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.relu(self.conv(x))


class mobilenetv2_4_tiny(nn.Module):
    def __init__(self, ):
        super(mobilenetv2_4_tiny, self).__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, groups=1),   # sum:32->prune:1
            # self.layer1
            InvertedResidual_Quantization_Friendly(32, 16, 1, 1),  # 1## sum:16->prune:0
            # self.layer2
            InvertedResidual_Quantization_Friendly(16, 24, 2, 4),  # 2 # 4# sum:64->prune:0# sum:24->prune:0
            InvertedResidual_Quantization_Friendly(24, 24, 1, 4),  # 3  # 4# sum:96->prune:0# sum:24->prune:0
            # self.layer3
            InvertedResidual_Quantization_Friendly(24, 32, 2, 4),  # 4 64->32 # sum:128->prune:23# sum:32->prune:0
            InvertedResidual_Quantization_Friendly(32, 32, 1, 4),  # 5  # # sum:128->prune:12# sum:32->prune:1
            InvertedResidual_Quantization_Friendly(32, 32, 1, 3),  # 6 # # sum:128->prune:21# sum:64->prune:39
            # self.layer4
            InvertedResidual_Quantization_Friendly(32, 48, 2, 4),  # 7 32->16 ## sum:128->prune:8# sum:64->prune:9
            InvertedResidual_Quantization_Friendly(48, 48, 1, 3),  # 8  # sum:256->prune:168# sum:64->prune:36
            InvertedResidual_Quantization_Friendly(48, 48, 1, 3),  # 9  # sum:256->prune:93# sum:64->prune:9
            InvertedResidual_Quantization_Friendly(48, 48, 1, 3),  # 10  # sum:256->prune:91# sum:64->prune:4
            #
            ConvBNReLU(48, 24, 3, 1, 1),  # 11# sum:32->prune:0
            ConvTranspose2d(24, 24, 4, 2, 1, 0, 24),  # 12 # 16->32
            Concat(1),  # 13  # cat[6,12]
            InvertedResidual_Quantization_Friendly(32 + 24, 56, 1, 2),  # 14 # 32->32# # sum:128->prune:25# sum:64->prune:6
        )

    def forward(self, x):
        x = self.features(x)
        return x


def corner_point_mobilenetv2_4_tiny(pretrained=False):
    model = mobilenetv2_4_tiny()
    return model


class mobilenetv2_4(nn.Module):
    def __init__(self, ):
        super(mobilenetv2_4, self).__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, groups=1),   # sum:32->prune:1
            # self.layer1
            InvertedResidual_Quantization_Friendly(32, 16, 1, 1),  # 1## sum:16->prune:0
            # self.layer2
            InvertedResidual_Quantization_Friendly(16, 24, 2, 4),  # 2 # 4# sum:64->prune:0# sum:24->prune:0
            InvertedResidual_Quantization_Friendly(24, 24, 1, 4),  # 3  # 4# sum:96->prune:0# sum:24->prune:0
            # self.layer3
            InvertedResidual_Quantization_Friendly(24, 32, 2, 4),  # 4 64->32 # sum:128->prune:23# sum:32->prune:0
            InvertedResidual_Quantization_Friendly(32, 32, 1, 4),  # 5  # # sum:128->prune:12# sum:32->prune:1
            InvertedResidual_Quantization_Friendly(32, 32, 1, 4),  # 6 # # sum:128->prune:21# sum:64->prune:39
            # self.layer4
            InvertedResidual_Quantization_Friendly(32, 64, 2, 4),  # 7 32->16 ## sum:128->prune:8# sum:64->prune:9
            InvertedResidual_Quantization_Friendly(64, 64, 1, 4),  # 8  # sum:256->prune:168# sum:64->prune:36
            InvertedResidual_Quantization_Friendly(64, 64, 1, 4),  # 9  # sum:256->prune:93# sum:64->prune:9
            InvertedResidual_Quantization_Friendly(64, 64, 1, 4),  # 10  # sum:256->prune:91# sum:64->prune:4
            #
            ConvBNReLU(64, 32, 3, 1, 1),  # 11# sum:32->prune:0
            ConvTranspose2d(32, 32, 4, 2, 1, 0, 32),  # 12 # 16->32
            Concat(1),  # 13  # cat[6,12]
            InvertedResidual_Quantization_Friendly(32 + 32, 64, 1, 2),  # 14 # 32->32# # sum:128->prune:25# sum:64->prune:6
        )

    def forward(self, x):
        x = self.features(x)
        return x


def corner_point_mobilenetv2_4(pretrained=False):
    model = mobilenetv2_4()
    return model


class mobilenetv2_4_advance(nn.Module):
    def __init__(self, ):
        super(mobilenetv2_4_advance, self).__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, groups=1),
            # self.layer1
            InvertedResidual_Quantization_Friendly(32, 16, 1, 1),  # 1#
            # self.layer2
            InvertedResidual_Quantization_Friendly(16, 24, 2, 4),  # 2 # 4
            InvertedResidual_Quantization_Friendly(24, 24, 1, 4),  # 3  # 4
            # self.layer3
            InvertedResidual_Quantization_Friendly(24, 32, 2, 4),  # 4 64->32 #3
            InvertedResidual_Quantization_Friendly(32, 32, 1, 4),  # 5  # 3
            InvertedResidual_Quantization_Friendly(32, 32, 1, 4),  # 6 # 3
            # self.layer4
            InvertedResidual_Quantization_Friendly(32, 64, 2, 4),  # 7 32->16 #
            InvertedResidual_Quantization_Friendly(64, 64, 1, 4),  # 8  #
            InvertedResidual_Quantization_Friendly(64, 64, 1, 4),  # 9  #
            InvertedResidual_Quantization_Friendly(64, 64, 1, 4),  # 10  #
            #
            ConvBNReLU(64, 32, 3, 1, 1),  # 11
            ConvTranspose2d(32, 32, 4, 2, 1, 0, 32),  # 12 # 16->32
            Concat(1),  # 13 # cat[6,12]
            InvertedResidual_Quantization_Friendly(32 + 32, 64, 1, 2),  # 14 # 32->32
            #
            ConvBNReLU(64, 32, 3, 2, 1),  # 15 # 32->16
            Concat(1),  # 16  # cat[10, 15]
            InvertedResidual_Quantization_Friendly(64 + 32, 64, 1, 2),  # 17 # 16->16

        )

    def forward(self, x):
        x = self.features(x)
        return x


def corner_point_mobilenetv2_4_advance(pretrained=False):
    model = mobilenetv2_4_advance()
    return model


MobileNetV2_4_2 = [
    [19, 20, 22, 23],
    # ## -------------backbone-------------
    [-1, ConvBNReLU1, [3, 32, 3, 2, 1, 1]],  # 0 256->128
    # self.layer1
    [-1, InvertedResidual_Quantization_Friendly, [32, 16, 1, 1]],  # 1#
    # self.layer2
    [-1, InvertedResidual_Quantization_Friendly, [16, 24, 2, 4]],  # 2 128->64
    [-1, InvertedResidual_Quantization_Friendly, [24, 24, 1, 4]],  # 3
    # self.layer3
    [-1, InvertedResidual_Quantization_Friendly, [24, 32, 2, 4]],  # 4 64->32
    [-1, InvertedResidual_Quantization_Friendly, [32, 32, 1, 4]],  # 5
    [-1, InvertedResidual_Quantization_Friendly, [32, 32, 1, 4]],  # 6
    # self.layer4
    [-1, InvertedResidual_Quantization_Friendly, [32, 64, 2, 4]],  # 7 32->16
    [-1, InvertedResidual_Quantization_Friendly, [64, 64, 1, 4]],  # 8
    [-1, InvertedResidual_Quantization_Friendly, [64, 64, 1, 4]],  # 9
    [-1, InvertedResidual_Quantization_Friendly, [64, 64, 1, 4]],  # 10
    # -------------neck--------------
    [-1, ConvBNReLU1, [64, 64, 3, 1, 1, 1]],  # 11
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64]],  # 12 16->32
    [[-1, 6], Concat1, [1]],  # 13 32->32
    [-1, InvertedResidual_Quantization_Friendly, [64+32, 64, 1, 2]],  # 14  branch0
    #
    [-1, ConvBNReLU1, [64, 32, 3, 2, 1, 1]],  # 15 32->16
    [[-1, 10], Concat1, [1]],  # 16
    [-1, InvertedResidual_Quantization_Friendly, [32+64, 64, 1, 2]],  # 17 branch1
    # -------------extras--------------
    [14, InvertedResidual_Quantization_Friendly, [64, 256, 1, 2]],  # 18
    [-1, Conv, [256, 4*4, 3, 1, 1, 1]],   # 19
    [18, Conv, [256, 4*6, 3, 1, 1, 1]],   # 20

    [17, InvertedResidual_Quantization_Friendly, [64, 256, 1, 2]],  # 21
    [-1, Conv, [256, 4*4, 3, 1, 1, 1]],  # 22
    [21, Conv, [256, 4*6, 3, 1, 1, 1]],  # 23
]

MobileNetV2_4_2_prune = [
    [19, 20, 22, 23],
    # ## -------------backbone-------------
    [-1, ConvBNReLU1, [3, 32, 3, 2, 1, 1]],  # 0 256->128
    # self.layer1
    [-1, InvertedResidual_prune, [32, 16, 1, 1]],  # 1#
    # self.layer2
    [-1, InvertedResidual_prune, [16, 24, 2, 4]],  # 2 128->64
    [-1, InvertedResidual_prune, [24, 24, 1, 4]],  # 3
    # self.layer3
    [-1, InvertedResidual_prune, [24, 32, 2, 4]],  # 4 64->32
    [-1, InvertedResidual_prune, [32, 32, 1, 4]],  # 5
    [-1, InvertedResidual_prune, [32, 32, 1, 4]],  # 6
    # self.layer4
    [-1, InvertedResidual_prune, [32, 64, 2, 4]],  # 7 32->16
    [-1, InvertedResidual_prune, [64, 64, 1, 4]],  # 8
    [-1, InvertedResidual_prune, [64, 64, 1, 4]],  # 9
    [-1, InvertedResidual_prune, [64, 64, 1, 4]],  # 10
    # -------------neck--------------
    [-1, ConvBNReLU1, [64, 64, 3, 1, 1, 1]],  # 11
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64]],  # 12 16->32
    [[-1, 6], Concat1, [1]],  # 13 32->32
    [-1, InvertedResidual_prune, [64+32, 64, 1, 2]],  # 14  branch0
    #
    [-1, ConvBNReLU1, [64, 32, 3, 2, 1, 1]],  # 15 32->16
    [[-1, 10], Concat1, [1]],  # 16
    [-1, InvertedResidual_prune, [32+64, 64, 1, 2]],  # 17 branch1
    # -------------extras--------------
    [14, InvertedResidual_prune, [64, 256, 1, 2]],  # 18
    [-1, Conv, [256, 4*4, 3, 1, 1, 1]],   # 19
    [18, Conv, [256, 4*6, 3, 1, 1, 1]],   # 20

    [17, InvertedResidual_prune, [64, 256, 1, 2]],  # 21
    [-1, Conv, [256, 4*4, 3, 1, 1, 1]],  # 22
    [21, Conv, [256, 4*6, 3, 1, 1, 1]],  # 23
]


MobileNetV2_4_2_test = [
    [19, 20, 22, 23],
    # ## -------------backbone-------------
    [-1, ConvBNReLU1, [3, 32, 3, 2, 1, 1]],  # 0 256->128
    # self.layer1
    [-1, InvertedResidual_Quantization_Friendly, [32, 16, 1, 1]],  # 1#
    # self.layer2
    [-1, InvertedResidual_Quantization_Friendly, [16, 24, 2, 4]],  # 2 128->64
    [-1, InvertedResidual_Quantization_Friendly, [24, 24, 1, 4]],  # 3
    # self.layer3
    [-1, InvertedResidual_Quantization_Friendly, [24, 32, 2, 4]],  # 4 64->32
    [-1, InvertedResidual_Quantization_Friendly, [32, 32, 1, 4]],  # 5
    [-1, InvertedResidual_Quantization_Friendly, [32, 32, 1, 4]],  # 6
    # self.layer4
    [-1, InvertedResidual_Quantization_Friendly, [32, 64, 2, 4]],  # 7 32->16
    [-1, InvertedResidual_Quantization_Friendly, [64, 64, 1, 4]],  # 8
    [-1, InvertedResidual_Quantization_Friendly, [64, 64, 1, 4]],  # 9
    [-1, InvertedResidual_Quantization_Friendly, [64, 64, 1, 4]],  # 10
    # -------------neck--------------
    [-1, ConvBNReLU1, [64, 64, 3, 1, 1, 1]],  # 11
    [-1, ConvTranspose2d, [64, 64, 4, 2, 1, 0, 64]],  # 12 16->32
    [[-1, 6], Concat1, [1]],  # 13 32->32
    [-1, InvertedResidual_Quantization_Friendly, [64+32, 64, 1, 2]],  # 14  branch0
    #
    [-1, ConvBNReLU1, [64, 32, 3, 2, 1, 1]],  # 15 32->16
    [[-1, 10], Concat1, [1]],  # 16
    [-1, InvertedResidual_Quantization_Friendly, [32+64, 64, 1, 2]],  # 17 branch1
    # -------------extras--------------
    [14, InvertedResidual_Quantization_Friendly, [64, 128, 1, 2]],  # 18
    [-1, Conv, [128, 4*4, 3, 1, 1, 1]],   # 19 第一个输出头，输出预测坐标 坐标4个值
    [18, Conv, [128, 4*6, 3, 1, 1, 1]],   # 20 第一个输出头，输出预测类别5+1 所有类别+背景类

    [17, InvertedResidual_Quantization_Friendly, [64, 128, 1, 2]],  # 21
    [-1, Conv, [128, 4*4, 3, 1, 1, 1]],  # 22 第二个输出头，输出预测坐标 坐标4个值
    [21, Conv, [128, 4*6, 3, 1, 1, 1]],  # 23 第二个输出头，输出预测类别5+1 所有类别+背景类
]



class MobileNetV2(nn.Module):
    def __init__(self, compress=1, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32

        if inverted_residual_setting is None:
            if compress == 1:
                inverted_residual_setting = [
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],  # 256->128
                    [6, 32, 3, 2],  # 128->64
                    [6, 64, 4, 2],  # 64->32
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],  # 32->16
                    [6, 320, 1, 1],
                ]
                last_channel = 1280
            elif compress == 0.5:
                inverted_residual_setting = [
                    [1, 8, 1, 1],
                    [4, 12, 2, 2],
                    [4, 16, 3, 2],
                    [4, 32, 4, 2],
                    [4, 48, 3, 1],
                    [4, 80, 3, 2],
                    [4, 160, 1, 1],
                ]
                last_channel = 640
            elif compress == 0.25:
                inverted_residual_setting = [
                    [1, 8, 1, 1],
                    [6, 8, 2, 2],
                    [6, 8, 3, 2],
                    [6, 16, 4, 2],
                    [6, 24, 3, 1],
                    [6, 40, 3, 2],
                    [6, 80, 1, 1],
                ]
                last_channel = 320

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=False, compress=1, **kwargs):
    model = MobileNetV2(compress, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth', model_dir="./model_data",
                                              progress=True)
        model.load_state_dict(state_dict)
    del model.classifier
    return model


if __name__ == "__main__":
    net = mobilenet_v2(pretrained=False, compress=0.5)
    for i, layer in enumerate(net.features):
        print(i, layer)