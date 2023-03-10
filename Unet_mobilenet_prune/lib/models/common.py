import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..core.general import non_max_suppression, non_max_suppression_export


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ConvBNReLU(nn.Module):
    # Standard convolution
    def __init__(self, inp, oup, kernel_size=3, stride=1, padding=0, groups=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

#  add mask
class ConvBNReLU_mask(nn.Module):
    # Standard convolution
    def __init__(self, inp, oup, kernel=3, stride=1, padding=0, group=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvBNReLU_mask, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel, stride, padding, groups=group, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.mask = nn.Conv2d(oup, oup, 1, groups=oup, bias=False)
        self.act = nn.ReLU(inplace=True)
        nn.init.ones_(self.mask.weight)

    def forward(self, x):
        return self.act(self.mask(self.bn(self.conv(x))))



class ConvBN(nn.Module):
    # Standard convolution
    def __init__(self, inp, oup, kernel=3, stride=1, padding=0, group=1, bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel, stride, padding, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(oup)

    def forward(self, x):
        return self.bn(self.conv(x))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=0.25):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            # layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            layers.extend([                         #kernel_size=3, stride=1, groups=1
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
        layers.extend([
            # dw
            # ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return ConvBNReLU(c1, c2, k, s, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = ConvBNReLU(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class DepthSeperabelConv2d(nn.Module):
    """
    DepthSeperable Convolution 2d with residual connection
    """

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, act=True):
        super(DepthSeperabelConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, groups=inplanes, padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(inplanes)
        )
        # self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, groups=inplanes, padding=1, bias=False)
        # self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.pointwise = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # residual = x

        out = self.depthwise(x)
        out = self.act(out)
        out = self.pointwise(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.act(out)

        return out


class SharpenConv(nn.Module):
    # SharpenConv convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SharpenConv, self).__init__()
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        kenel_weight = np.vstack([sobel_kernel] * c2 * c1).reshape(c2, c1, 3, 3)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv.weight.data = torch.from_numpy(kenel_weight)
        self.conv.weight.requires_grad = False
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock_mask(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(ResBlock_mask, self).__init__()
        assert mid_planes > in_planes

        self.in_planes = in_planes
        self.mid_planes = mid_planes - out_planes + in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, self.mid_planes - in_planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes - in_planes)
        self.mask1 = nn.Conv2d(self.mid_planes - in_planes, self.mid_planes - in_planes, 1,
                               groups=self.mid_planes - in_planes, bias=False)

        self.conv2 = nn.Conv2d(self.mid_planes - in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.mask2 = nn.Conv2d(out_planes, out_planes, 1, groups=out_planes, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if self.in_planes != self.out_planes or self.stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes))

        self.mask_res = nn.Sequential(*[nn.Conv2d(self.in_planes, self.in_planes, 1, groups=self.in_planes, bias=False),
                                        nn.ReLU(inplace=True)])

        self.running1 = nn.BatchNorm2d(in_planes, affine=False)
        self.running2 = nn.BatchNorm2d(out_planes, affine=False)

        nn.init.ones_(self.mask1.weight)
        nn.init.ones_(self.mask2.weight)
        nn.init.ones_(self.mask_res[0].weight)

    def forward(self, x):
        if self.in_planes == self.out_planes and self.stride == 1:
            self.running1(x)
        # print(x)
        # print(self.mask_res(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mask1(out)
        out = self.relu(out)
        # print(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.downsample(self.mask_res(x)) + out
        self.running2(out)
        out = self.mask2(out)
        out = self.relu(out)
        # print(out.shape)
        # print(out)

        return out

    def deploy(self, merge_bn=False):
        idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=self.stride, padding=1,
                            bias=False).eval()
        idbn1 = nn.BatchNorm2d(self.mid_planes).eval()
        # dirac初始化矩阵，中心为 1 的矩阵
        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        # dirac初始化矩阵 BN 层权重计算
        bn_var_sqrt = torch.sqrt(self.running1.running_var + self.running1.eps)
        # 对于cat后半部分的BN作用进行赋值，主要还是 running_mean，running_var，weight，bias
        # dirac初始化矩阵 weight 的赋值
        idbn1.weight.data[:self.in_planes] = bn_var_sqrt
        # dirac初始化矩阵 bias 的赋值
        idbn1.bias.data[:self.in_planes] = self.running1.running_mean
        # dirac初始化矩阵 running_mean 的赋值
        idbn1.running_mean.data[:self.in_planes] = self.running1.running_mean
        # dirac初始化矩阵 running_var 的赋值
        idbn1.running_var.data[:self.in_planes] = self.running1.running_var

        # 原始特征矩阵 卷积 weight 的赋值
        idconv1.weight.data[self.in_planes:] = self.conv1.weight.data
        # 原始特征矩阵 BN层 weight 的赋值
        idbn1.weight.data[self.in_planes:] = self.bn1.weight.data
        # 原始特征矩阵 BN层 bias 的赋值
        idbn1.bias.data[self.in_planes:] = self.bn1.bias.data
        # 原始特征矩阵 BN层 running_mean 的赋值
        idbn1.running_mean.data[self.in_planes:] = self.bn1.running_mean
        # 原始特征矩阵 BN层 running_var 的赋值
        idbn1.running_var.data[self.in_planes:] = self.bn1.running_var
        # init mask_res mask to mask1
        mask1 = nn.Conv2d(self.mid_planes, self.mid_planes, 1, groups=self.mid_planes, bias=False)
        mask1.weight.data[:self.in_planes] = self.mask_res[0].weight.data * (self.mask_res[0].weight.data > 0)
        mask1.weight.data[self.in_planes:] = self.mask1.weight.data
        idbn1.weight.data *= mask1.weight.data.reshape(-1)
        idbn1.bias.data *= mask1.weight.data.reshape(-1)
        # 实例化一个 cat channel 后的卷积
        # init idconv2
        idconv2 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2 = nn.BatchNorm2d(self.out_planes).eval()

        # 进行 Conv2 的操作
        downsample_bias = 0
        if self.in_planes == self.out_planes:
            # Channel number 不变的话，直接对idconv2的后部分进行dirac初始化
            nn.init.dirac_(idconv2.weight.data[:, :self.in_planes])
        else:
            # Channel number 变化的话，这时需要考虑downsample的weight，bias，然后进行融合
            idconv2.weight.data[:, :self.in_planes], downsample_bias = self.fuse(
                F.pad(self.downsample[0].weight.data, [1, 1, 1, 1]), self.downsample[1].running_mean,
                self.downsample[1].running_var, self.downsample[1].weight, self.downsample[1].bias,
                self.downsample[1].eps)

        idconv2.weight.data[:, self.in_planes:], bias = self.fuse(self.conv2.weight, self.bn2.running_mean,
                                                                  self.bn2.running_var, self.bn2.weight, self.bn2.bias,
                                                                  self.bn2.eps)

        bn_var_sqrt = torch.sqrt(self.running2.running_var + self.running2.eps)
        idbn2.weight.data = bn_var_sqrt
        idbn2.bias.data = self.running2.running_mean
        idbn2.running_mean.data = self.running2.running_mean + bias + downsample_bias
        idbn2.running_var.data = self.running2.running_var
        idbn2.weight.data *= self.mask2.weight.data.reshape(-1)
        idbn2.bias.data *= self.mask2.weight.data.reshape(-1)
        return [idconv1, idbn1, nn.ReLU(True), idconv2, idbn2, nn.ReLU(True)]

    def fuse(self, conv_w, bn_rm, bn_rv, bn_w, bn_b, eps):
        bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        conv_b = bn_rm * bn_var_rsqrt * bn_w - bn_b
        return conv_w, conv_b


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBNReLU(c1, c_, 1, 1)
        self.cv2 = ConvBNReLU(c_, c2, 3, 1, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_mask(nn.Module):
    # Standard bottleneck
    def __init__(self, in_planes, out_planes, shortcut=True, expand_ratio=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck_mask, self).__init__()
        self.add = shortcut and in_planes == out_planes
        self.in_planes = in_planes
        self.mid_planes = int(out_planes * expand_ratio)  # hidden channels
        self.out_planes = out_planes

        self.mask_res1 = nn.Sequential(*[nn.Conv2d(self.in_planes, self.in_planes, 1, groups=self.in_planes, bias=False),
                                        nn.ReLU(inplace=True)])
        self.mask_res2 = nn.Sequential(*[nn.Conv2d(self.in_planes, self.in_planes, 1, groups=self.in_planes, bias=False),
                                        nn.ReLU(inplace=True)])

        self.conv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes)
        self.mask1 = nn.Conv2d(self.mid_planes, self.mid_planes, 1, groups=self.mid_planes, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_planes)
        self.mask2 = nn.Conv2d(self.out_planes, self.out_planes, 1, groups=self.out_planes, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.mask3 = nn.Conv2d(self.out_planes, self.out_planes, 1, groups=self.out_planes, bias=False)

        self.running1 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.running2 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.running3 = nn.BatchNorm2d(self.out_planes, affine=False)
        nn.init.ones_(self.mask1.weight)
        nn.init.ones_(self.mask2.weight)
        nn.init.ones_(self.mask3.weight)
        nn.init.ones_(self.mask_res1[0].weight)
        nn.init.ones_(self.mask_res2[0].weight)

    def forward(self, x):
        # x + self.cv2(self.cv1(x))
        if self.add:
            # print(x)
            self.running1(x)
            branch1_1 = self.mask_res1(x)
            # print(branch1_1)
            self.running2(branch1_1)
            branch2_1 = self.mask_res2(branch1_1)
            # print(branch2_1)
            branch1_2 = self.relu1(self.mask1(self.bn1(self.conv1(x))))
            # print(branch1_2)
            branch2_2 = self.relu2(self.mask2(self.bn2(self.conv2(branch1_2))))
            # print(branch2_2)
            out = branch2_1 + branch2_2
            # self.running3(out)
            # print(out)
            out = self.mask3(out)
            # print(out)
            return out
        else:
            out = self.relu1(self.bn1(self.conv1(x)))
            # print(out)
            out = self.relu2(self.bn2(self.conv2(out)))
            return out

    def deploy(self, merge_bn=False):
        # feature map的running mean和running variance分别为miu和sigma^2,
        # w=np.sqrt(sigma^2+np.exs),b=miu
        idconv1 = nn.Conv2d(self.in_planes, self.in_planes + self.mid_planes, kernel_size=1, stride=1, padding=0, bias=False).eval()
        idbn1 = nn.BatchNorm2d(self.in_planes + self.mid_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        bn_var_sqrt1 = torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn1.weight.data[:self.in_planes] = bn_var_sqrt1
        idbn1.bias.data[:self.in_planes] = self.running1.running_mean
        idbn1.running_mean.data[:self.in_planes] = self.running1.running_mean
        idbn1.running_var.data[:self.in_planes] = self.running1.running_var
        # init conv1 to idconv1
        idconv1.weight.data[self.in_planes:] = self.conv1.weight.data
        idbn1.weight.data[self.in_planes:] = self.bn1.weight.data
        idbn1.bias.data[self.in_planes:] = self.bn1.bias.data
        idbn1.running_mean.data[self.in_planes:] = self.bn1.running_mean
        idbn1.running_var.data[self.in_planes:] = self.bn1.running_var
        # sparsity mask
        mask_res1_weight_data = (self.mask_res1[0].weight.data * (self.mask_res1[0].weight.data > 0)).reshape(-1)
        mask1_weight_data = self.mask1.weight.data.reshape(-1)
        weight_data = torch.cat((mask_res1_weight_data, mask1_weight_data), 0)
        idbn1.weight.data *= weight_data
        idbn1.bias.data *= weight_data

        # conv2
        idconv2 = nn.Conv2d(self.in_planes + self.mid_planes, self.in_planes + self.out_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2 = nn.BatchNorm2d(self.in_planes + self.out_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        bn_var_sqrt2 = torch.sqrt(self.running2.running_var + self.running2.eps)
        idbn2.weight.data[:self.in_planes] = bn_var_sqrt2
        idbn2.bias.data[:self.in_planes] = self.running2.running_mean
        idbn2.running_mean.data[:self.in_planes] = self.running2.running_mean
        idbn2.running_var.data[:self.in_planes] = self.running2.running_var
        nn.init.zeros_(idconv2.weight.data)
        nn.init.dirac_(idconv2.weight.data[:self.in_planes, :self.in_planes])  # , :self.in_planes#
        # init conv2 to idconv2
        idconv2.weight.data[self.in_planes:, self.in_planes:] = self.conv2.weight.data#, self.in_planes:
        idbn2.weight.data[self.in_planes:] = self.bn2.weight.data
        idbn2.bias.data[self.in_planes:] = self.bn2.bias.data
        idbn2.running_mean.data[self.in_planes:] = self.bn2.running_mean
        idbn2.running_var.data[self.in_planes:] = self.bn2.running_var
        # sparsity mask
        mask_res2_weight_data = (self.mask_res2[0].weight.data * (self.mask_res2[0].weight.data > 0)).reshape(-1)
        mask2_weight_data = self.mask2.weight.data.reshape(-1)
        weight_data = torch.cat((mask_res2_weight_data, mask2_weight_data), 0)
        idbn2.weight.data *= weight_data
        idbn2.bias.data *= weight_data

        # init idconv3
        idconv3 = nn.Conv2d(self.in_planes + self.out_planes, self.out_planes, kernel_size=1, stride=1, padding=0, bias=False).eval()
        idbn3 = nn.BatchNorm2d(self.out_planes).eval()
        nn.init.dirac_(idconv3.weight.data[:, :self.in_planes])
        nn.init.dirac_(idconv3.weight.data[:, self.in_planes:])
        # bn_var_sqrt3 = torch.sqrt(self.running3.running_var + self.running3.eps)
        # idbn3.weight.data = bn_var_sqrt3
        # idbn3.bias.data = self.running3.running_mean
        # idbn3.running_mean.data = self.running3.running_mean
        # idbn3.running_var.data = self.running3.running_var
        idbn3.weight.data *= self.mask3.weight.data.reshape(-1)
        idbn3.bias.data *= self.mask3.weight.data.reshape(-1)

        return [idconv1, idbn1, nn.ReLU(True), idconv2, idbn2, nn.ReLU(True), idconv3, idbn3]


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBNReLU(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = ConvBNReLU(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.ReLU(inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class VoV(nn.Module):
    # VoV yolov4-tiny
    def __init__(self, in_planes, out_planes, expand_ratio=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(VoV, self).__init__()
        self.in_planes = in_planes
        self.mid_planes = int(out_planes * expand_ratio)  # hidden channels
        self.out_planes = out_planes
        #     self.cv1 = ConvBNReLU(c1, c_//2, 3, 1)
        self.conv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes)
        self.relu1 = nn.ReLU(inplace=True)
        #     self.cv2 = ConvBNReLU(c_//2, c_//2, 3, 1)
        self.conv2 = nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_planes)
        self.relu2 = nn.ReLU(inplace=True)
        #     self.cv3 = ConvBNReLU(c_, c2, 1, 1)
        self.conv3 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_planes)
        self.relu3 = nn.ReLU(inplace=True)

        self.running1 = nn.BatchNorm2d(self.mid_planes, affine=False)

    def forward(self, x):
        # print(x)
        x1 = self.relu1(self.bn1(self.conv1(x)))   # x1 = self.cv1(x)
        # print(x1)
        self.running1(x1)
        x2 = self.relu2(self.bn2(self.conv2(x1)))   # x2 = self.cv2(x1)
        # print(x2)
        x3 = torch.cat((x1, x2), dim=1)   # x3 = torch.cat((x1, x2), dim=1)
        # print(x3)
        out = self.relu3(self.bn3(self.conv3(x3)))   #self.cv3(x3)
        # print(out)
        return out

    def deploy(self):
        # feature map的running mean和running variance分别为miu和sigma^2,
        # w=np.sqrt(sigma^2+np.exs),b=miu
        # conv1
        idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn1 = nn.BatchNorm2d(self.mid_planes).eval()
        # init conv1 to idconv1
        idconv1.weight.data = self.conv1.weight.data
        idbn1.weight.data = self.bn1.weight.data
        idbn1.bias.data = self.bn1.bias.data
        idbn1.running_mean.data = self.bn1.running_mean
        idbn1.running_var.data = self.bn1.running_var

        # conv2
        idconv2 = nn.Conv2d(self.mid_planes, self.mid_planes+self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2 = nn.BatchNorm2d(self.mid_planes + self.mid_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        bn_var_sqrt2 = torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn2.weight.data[:self.mid_planes] = bn_var_sqrt2
        idbn2.bias.data[:self.mid_planes] = self.running1.running_mean
        idbn2.running_mean.data[:self.mid_planes] = self.running1.running_mean
        idbn2.running_var.data[:self.mid_planes] = self.running1.running_var

        nn.init.zeros_(idconv2.weight.data)
        nn.init.dirac_(idconv2.weight.data[:self.mid_planes])  # , :self.in_planes#

        # init conv2 to idconv2
        idconv2.weight.data[self.mid_planes:, ] = self.conv2.weight.data  # , self.in_planes:
        idbn2.weight.data[self.mid_planes:] = self.bn2.weight.data
        idbn2.bias.data[self.mid_planes:] = self.bn2.bias.data
        idbn2.running_mean.data[self.mid_planes:] = self.bn2.running_mean
        idbn2.running_var.data[self.mid_planes:] = self.bn2.running_var

        # conv3
        idconv3 = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=1, stride=1, padding=0, bias=False).eval()
        idbn3 = nn.BatchNorm2d(self.in_planes).eval()
        # init conv3 to idconv3
        idconv3.weight.data = self.conv3.weight.data  # , self.in_planes:
        idbn3.weight.data = self.bn3.weight.data
        idbn3.bias.data = self.bn3.bias.data
        idbn3.running_mean.data = self.bn3.running_mean
        idbn3.running_var.data = self.bn3.running_var

        return [idconv1, idbn1, nn.ReLU(True), idconv2, idbn2, nn.ReLU(True), idconv3, idbn3, nn.ReLU(True)]


class VoV_mask(nn.Module):
    # VoV yolov4-tiny
    def __init__(self, in_planes, out_planes, expand_ratio=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(VoV_mask, self).__init__()
        self.in_planes = in_planes
        self.mid_planes = int(out_planes * expand_ratio)  # hidden channels
        self.out_planes = out_planes

        self.conv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes)
        self.mask1 = nn.Conv2d(self.mid_planes, self.mid_planes, 1, groups=self.mid_planes, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.mask_res = nn.Sequential(*[nn.Conv2d(self.mid_planes, self.mid_planes, 1, groups=self.mid_planes, bias=False),
                                        nn.ReLU(inplace=True)])

        self.conv2 = nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_planes)
        self.mask2 = nn.Conv2d(self.mid_planes, self.mid_planes, 1, groups=self.mid_planes, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_planes)
        self.mask3 = nn.Conv2d(self.out_planes, self.out_planes, 1, groups=self.out_planes, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.running1 = nn.BatchNorm2d(self.mid_planes, affine=False)
        nn.init.ones_(self.mask1.weight)
        nn.init.ones_(self.mask2.weight)
        nn.init.ones_(self.mask3.weight)
        nn.init.ones_(self.mask_res[0].weight)

    def forward(self, x):
        # print(np.around(x, 4))
        x1 = self.relu1(self.mask1(self.bn1(self.conv1(x))))   # x1 = self.cv1(x)
        # print(np.around(x1, 4))
        self.running1(x1)
        x2_1 = self.mask_res(x1)
        x2_2 = self.relu2(self.mask2(self.bn2(self.conv2(x1))))   # x2 = self.cv2(x1)
        # print(np.around(x2_2, 4))
        x3 = torch.cat((x2_1, x2_2), dim=1)   # x3 = torch.cat((x1, x2), dim=1)
        # print(np.around(x3, 4))
        out = self.relu3(self.mask3(self.bn3(self.conv3(x3))))   #self.cv3(x3) # error
        # print(np.around(out, 4))
        return out

    def deploy(self):
        # feature map的running mean和running variance分别为miu和sigma^2,
        # w=np.sqrt(sigma^2+np.exs),b=miu
        # conv1
        idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn1 = nn.BatchNorm2d(self.mid_planes).eval()
        # init conv1 to idconv1
        idconv1.weight.data = self.conv1.weight.data
        idbn1.weight.data = self.bn1.weight.data
        idbn1.bias.data = self.bn1.bias.data
        idbn1.running_mean.data = self.bn1.running_mean
        idbn1.running_var.data = self.bn1.running_var
        # sparsity mask
        idbn1.weight.data *= self.mask1.weight.data.reshape(-1)
        idbn1.bias.data *= self.mask1.weight.data.reshape(-1)

        # conv2
        idconv2 = nn.Conv2d(self.mid_planes, self.mid_planes+self.mid_planes,
                            kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2 = nn.BatchNorm2d(self.mid_planes+self.mid_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        bn_var_sqrt2 = torch.sqrt(self.running1.running_var+self.running1.eps)
        idbn2.weight.data[:self.mid_planes] = bn_var_sqrt2
        idbn2.bias.data[:self.mid_planes] = self.running1.running_mean
        idbn2.running_mean.data[:self.mid_planes] = self.running1.running_mean
        idbn2.running_var.data[:self.mid_planes] = self.running1.running_var
        nn.init.zeros_(idconv2.weight.data)
        nn.init.dirac_(idconv2.weight.data[:self.mid_planes])  # , :self.in_planes#
        # init conv2 to idconv2
        idconv2.weight.data[self.mid_planes:, ] = self.conv2.weight.data  # , self.in_planes:
        idbn2.weight.data[self.mid_planes:] = self.bn2.weight.data
        idbn2.bias.data[self.mid_planes:] = self.bn2.bias.data
        idbn2.running_mean.data[self.mid_planes:] = self.bn2.running_mean
        idbn2.running_var.data[self.mid_planes:] = self.bn2.running_var
        # init mask_res mask to mask1
        mask_res1_weight_data = (self.mask_res[0].weight.data * (self.mask_res[0].weight.data > 0)).reshape(-1)
        mask1_weight_data = self.mask2.weight.data.reshape(-1)
        weight_data = torch.cat((mask_res1_weight_data, mask1_weight_data), 0)
        idbn2.weight.data *= weight_data
        idbn2.bias.data *= weight_data

        # conv3
        idconv3 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=1, stride=1, padding=0, bias=False).eval()
        idbn3 = nn.BatchNorm2d(self.out_planes).eval()
        # init conv3 to idconv3
        idconv3.weight.data = self.conv3.weight.data  # , self.in_planes:
        idbn3.weight.data = self.bn3.weight.data
        idbn3.bias.data = self.bn3.bias.data
        idbn3.running_mean.data = self.bn3.running_mean
        idbn3.running_var.data = self.bn3.running_var
        # sparsity mask
        idbn3.weight.data *= self.mask3.weight.data.reshape(-1)
        idbn3.bias.data *= self.mask3.weight.data.reshape(-1)

        return [idconv1, idbn1, nn.ReLU(True), idconv2, idbn2, nn.ReLU(True), idconv3, idbn3, nn.ReLU(True)]


class VoVCSP(nn.Module):
    # VoVCSP yolov4-tiny
    def __init__(self, in_planes, out_planes, n=1, shortcut=True, g=1, expand_ratio=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(VoVCSP, self).__init__()
        self.in_planes = in_planes
        self.mid_planes = int(out_planes * expand_ratio)  # hidden channels
        self.out_planes = out_planes

        self.conv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_planes)
        self.relu3 = nn.ReLU(inplace=True)

        self.running1 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.running2 = nn.BatchNorm2d(self.mid_planes, affine=False)

    def forward(self, x):
        self.running1(x)
        # print(x)
        x1 = self.relu1(self.bn1(self.conv1(x)))
        # print(x1)
        self.running2(x1)
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        # print(x2)
        x3 = torch.cat((x1, x2), dim=1)
        # print(x3)
        x4 = self.relu3(self.bn3(self.conv3(x3)))
        # print(x4)
        out = torch.cat((x, x4), dim=1)
        # print(out)
        return out

    def deploy(self):
        idconv1 = nn.Conv2d(self.in_planes, self.in_planes+self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn1 = nn.BatchNorm2d(self.in_planes+self.mid_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        bn_var_sqrt1 = torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn1.weight.data[:self.in_planes] = bn_var_sqrt1
        idbn1.bias.data[:self.in_planes] = self.running1.running_mean
        idbn1.running_mean.data[:self.in_planes] = self.running1.running_mean
        idbn1.running_var.data[:self.in_planes] = self.running1.running_var
        # init conv1 to idconv1
        idconv1.weight.data[self.in_planes:] = self.conv1.weight.data
        idbn1.weight.data[self.in_planes:] = self.bn1.weight.data
        idbn1.bias.data[self.in_planes:] = self.bn1.bias.data
        idbn1.running_mean.data[self.in_planes:] = self.bn1.running_mean
        idbn1.running_var.data[self.in_planes:] = self.bn1.running_var

        # conv2
        idconv2 = nn.Conv2d(self.in_planes+self.mid_planes, self.in_planes+self.mid_planes+self.mid_planes,
                            kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2 = nn.BatchNorm2d(self.in_planes+self.mid_planes+self.mid_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        bn_var_sqrt1 = torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn2.weight.data[:self.in_planes] = bn_var_sqrt1
        idbn2.bias.data[:self.in_planes] = self.running1.running_mean
        idbn2.running_mean.data[:self.in_planes] = self.running1.running_mean
        idbn2.running_var.data[:self.in_planes] = self.running1.running_var

        bn_var_sqrt2 = torch.sqrt(self.running2.running_var + self.running2.eps)
        idbn2.weight.data[self.in_planes:self.in_planes+self.mid_planes] = bn_var_sqrt2
        idbn2.bias.data[self.in_planes:self.in_planes+self.mid_planes] = self.running2.running_mean
        idbn2.running_mean.data[self.in_planes:self.in_planes+self.mid_planes] = self.running2.running_mean
        idbn2.running_var.data[self.in_planes:self.in_planes+self.mid_planes] = self.running2.running_var

        nn.init.zeros_(idconv2.weight.data)
        nn.init.dirac_(idconv2.weight.data[:self.in_planes, :self.in_planes])  # , :self.in_planes#
        nn.init.dirac_(idconv2.weight.data[self.in_planes:self.in_planes+self.mid_planes, self.in_planes:self.in_planes+self.mid_planes])  # , :self.in_planes#
        # init conv2 to idconv2
        idconv2.weight.data[self.in_planes+self.mid_planes:, self.in_planes:] = self.conv2.weight.data  # , self.in_planes:
        idbn2.weight.data[self.in_planes+self.mid_planes:] = self.bn2.weight.data
        idbn2.bias.data[self.in_planes+self.mid_planes:] = self.bn2.bias.data
        idbn2.running_mean.data[self.in_planes+self.mid_planes:] = self.bn2.running_mean
        idbn2.running_var.data[self.in_planes+self.mid_planes:] = self.bn2.running_var

        # conv3
        idconv3 = nn.Conv2d(self.in_planes + self.mid_planes+self.mid_planes, self.in_planes + self.out_planes,
                            kernel_size=1, stride=1, padding=0, bias=False).eval()
        idbn3 = nn.BatchNorm2d(self.in_planes + self.out_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        bn_var_sqrt1 = torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn3.weight.data[:self.in_planes] = bn_var_sqrt1
        idbn3.bias.data[:self.in_planes] = self.running1.running_mean
        idbn3.running_mean.data[:self.in_planes] = self.running1.running_mean
        idbn3.running_var.data[:self.in_planes] = self.running1.running_var

        nn.init.zeros_(idconv3.weight.data)
        nn.init.dirac_(idconv3.weight.data[:self.in_planes])  # , :self.in_planes#
        # init conv3 to idconv3
        idconv3.weight.data[self.in_planes:, self.in_planes:] = self.conv3.weight.data  # , self.in_planes:
        idbn3.weight.data[self.in_planes:] = self.bn3.weight.data
        idbn3.bias.data[self.in_planes:] = self.bn3.bias.data
        idbn3.running_mean.data[self.in_planes:] = self.bn3.running_mean
        idbn3.running_var.data[self.in_planes:] = self.bn3.running_var

        return [idconv1, idbn1, nn.ReLU(True), idconv2, idbn2, nn.ReLU(True), idconv3, idbn3, nn.ReLU(True)]


class VoVCSP_mask(nn.Module):
    # VoVCSP yolov4-tiny
    def __init__(self, in_planes, out_planes, n=1, shortcut=True, g=1, expand_ratio=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(VoVCSP_mask, self).__init__()
        self.in_planes = in_planes
        self.mid_planes = int(out_planes * expand_ratio)  # hidden channels
        self.out_planes = out_planes

        self.running1_1 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.mask_res1_1 = nn.Sequential(*[nn.Conv2d(self.in_planes, self.in_planes, 1, groups=self.in_planes, bias=False),
                                        nn.ReLU(inplace=True)])
        self.running2_1 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.mask_res2_1 = nn.Sequential(*[nn.Conv2d(self.in_planes, self.in_planes, 1, groups=self.in_planes, bias=False),
                                        nn.ReLU(inplace=True)])
        self.running3_1 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.mask_res3_1 = nn.Sequential(*[nn.Conv2d(self.in_planes, self.in_planes, 1, groups=self.in_planes, bias=False),
                                        nn.ReLU(inplace=True)])
        self.running2_2 = nn.BatchNorm2d(self.mid_planes, affine=False)
        self.mask_res2_2 = nn.Sequential(*[nn.Conv2d(self.mid_planes, self.mid_planes, 1, groups=self.mid_planes, bias=False),
                                        nn.ReLU(inplace=True)])

        self.conv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes)
        self.mask1 = nn.Conv2d(self.mid_planes, self.mid_planes, 1, groups=self.mid_planes , bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_planes)
        self.mask2 = nn.Conv2d(self.mid_planes, self.mid_planes, 1, groups=self.mid_planes, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_planes)
        self.mask3 = nn.Conv2d(self.out_planes, self.out_planes, 1, groups=self.out_planes, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.running1_1 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.running2_1 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.running3_1 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.running2_2 = nn.BatchNorm2d(self.mid_planes, affine=False)

        nn.init.ones_(self.mask1.weight)
        nn.init.ones_(self.mask2.weight)
        nn.init.ones_(self.mask3.weight)
        nn.init.ones_(self.mask_res1_1[0].weight)
        nn.init.ones_(self.mask_res2_1[0].weight)
        nn.init.ones_(self.mask_res3_1[0].weight)
        nn.init.ones_(self.mask_res2_2[0].weight)

    def forward(self, x):
        # print(x)
        self.running1_1(x)
        branch1_1 = self.mask_res1_1(x)
        branch1_2 = self.relu1(self.mask1(self.bn1(self.conv1(x))))
        # print(torch.cat((branch1_1, branch1_2), dim=1))

        self.running2_1(branch1_1)
        branch2_1 = self.mask_res2_1(branch1_1)
        self.running2_2(branch1_2)
        branch2_2 = self.mask_res2_2(branch1_2)
        branch2_3 = self.relu2(self.mask2(self.bn2(self.conv2(branch1_2))))
        # print(torch.cat((branch2_1, branch2_2, branch2_3), dim=1))

        self.running3_1(branch2_1)
        branch3_1 = self.mask_res3_1(branch2_1)
        x3 = torch.cat((branch2_2, branch2_3), dim=1)
        # print(x3)
        branch3_2 = self.relu3(self.mask3(self.bn3(self.conv3(x3))))
        out = torch.cat((branch3_1, branch3_2), dim=1)
        # print(out)
        return out

    def deploy(self):
        idconv1 = nn.Conv2d(self.in_planes, self.in_planes+self.mid_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn1 = nn.BatchNorm2d(self.in_planes+self.mid_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        bn_var_sqrt1 = torch.sqrt(self.running1_1.running_var + self.running1_1.eps)
        idbn1.weight.data[:self.in_planes] = bn_var_sqrt1
        idbn1.bias.data[:self.in_planes] = self.running1_1.running_mean
        idbn1.running_mean.data[:self.in_planes] = self.running1_1.running_mean
        idbn1.running_var.data[:self.in_planes] = self.running1_1.running_var
        # init conv1 to idconv1
        idconv1.weight.data[self.in_planes:] = self.conv1.weight.data
        idbn1.weight.data[self.in_planes:] = self.bn1.weight.data
        idbn1.bias.data[self.in_planes:] = self.bn1.bias.data
        idbn1.running_mean.data[self.in_planes:] = self.bn1.running_mean
        idbn1.running_var.data[self.in_planes:] = self.bn1.running_var
        # sparsity mask
        mask_res1_weight_data = (self.mask_res1_1[0].weight.data * (self.mask_res1_1[0].weight.data > 0)).reshape(-1)
        mask1_weight_data = self.mask1.weight.data.reshape(-1)
        weight_data = torch.cat((mask_res1_weight_data, mask1_weight_data), 0)
        idbn1.weight.data *= weight_data
        idbn1.bias.data *= weight_data

        # conv2
        idconv2 = nn.Conv2d(self.in_planes+self.mid_planes, self.in_planes+self.mid_planes+self.mid_planes,
                            kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2 = nn.BatchNorm2d(self.in_planes+self.mid_planes+self.mid_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        bn_var_sqrt1 = torch.sqrt(self.running2_1.running_var + self.running2_1.eps)
        idbn2.weight.data[:self.in_planes] = bn_var_sqrt1
        idbn2.bias.data[:self.in_planes] = self.running2_1.running_mean
        idbn2.running_mean.data[:self.in_planes] = self.running2_1.running_mean
        idbn2.running_var.data[:self.in_planes] = self.running2_1.running_var

        bn_var_sqrt2 = torch.sqrt(self.running2_2.running_var + self.running2_2.eps)
        idbn2.weight.data[self.in_planes:self.in_planes+self.mid_planes] = bn_var_sqrt2
        idbn2.bias.data[self.in_planes:self.in_planes+self.mid_planes] = self.running2_2.running_mean
        idbn2.running_mean.data[self.in_planes:self.in_planes+self.mid_planes] = self.running2_2.running_mean
        idbn2.running_var.data[self.in_planes:self.in_planes+self.mid_planes] = self.running2_2.running_var

        nn.init.zeros_(idconv2.weight.data)
        nn.init.dirac_(idconv2.weight.data[:self.in_planes, :self.in_planes])  # , :self.in_planes#
        nn.init.dirac_(idconv2.weight.data[self.in_planes:self.in_planes+self.mid_planes, self.in_planes:self.in_planes+self.mid_planes])  # , :self.in_planes#
        # init conv2 to idconv2
        idconv2.weight.data[self.in_planes+self.mid_planes:, self.in_planes:] = self.conv2.weight.data  # , self.in_planes:
        idbn2.weight.data[self.in_planes+self.mid_planes:] = self.bn2.weight.data
        idbn2.bias.data[self.in_planes+self.mid_planes:] = self.bn2.bias.data
        idbn2.running_mean.data[self.in_planes+self.mid_planes:] = self.bn2.running_mean
        idbn2.running_var.data[self.in_planes+self.mid_planes:] = self.bn2.running_var
        # sparsity mask
        mask_res1_weight_data = (self.mask_res2_1[0].weight.data * (self.mask_res2_1[0].weight.data > 0)).reshape(-1)
        mask_res2_weight_data = (self.mask_res2_2[0].weight.data * (self.mask_res2_2[0].weight.data > 0)).reshape(-1)
        mask1_weight_data = self.mask2.weight.data.reshape(-1)
        weight_data = torch.cat((mask_res1_weight_data, mask_res2_weight_data, mask1_weight_data), 0)
        idbn2.weight.data *= weight_data
        idbn2.bias.data *= weight_data

        # conv3
        idconv3 = nn.Conv2d(self.in_planes + self.mid_planes+self.mid_planes, self.in_planes + self.out_planes,
                            kernel_size=1, stride=1, padding=0, bias=False).eval()
        idbn3 = nn.BatchNorm2d(self.in_planes + self.out_planes).eval()
        # init dirac_ kernel weight, bias, mean var to idconv1
        bn_var_sqrt1 = torch.sqrt(self.running3_1.running_var + self.running3_1.eps)
        idbn3.weight.data[:self.in_planes] = bn_var_sqrt1
        idbn3.bias.data[:self.in_planes] = self.running3_1.running_mean
        idbn3.running_mean.data[:self.in_planes] = self.running3_1.running_mean
        idbn3.running_var.data[:self.in_planes] = self.running3_1.running_var

        nn.init.zeros_(idconv3.weight.data)
        nn.init.dirac_(idconv3.weight.data[:self.in_planes])  # , :self.in_planes#
        # init conv3 to idconv3
        idconv3.weight.data[self.in_planes:, self.in_planes:] = self.conv3.weight.data  # , self.in_planes:
        idbn3.weight.data[self.in_planes:] = self.bn3.weight.data
        idbn3.bias.data[self.in_planes:] = self.bn3.bias.data
        idbn3.running_mean.data[self.in_planes:] = self.bn3.running_mean
        idbn3.running_var.data[self.in_planes:] = self.bn3.running_var
        mask_res1_weight_data = (self.mask_res3_1[0].weight.data * (self.mask_res3_1[0].weight.data > 0)).reshape(-1)
        mask1_weight_data = self.mask3.weight.data.reshape(-1)
        weight_data = torch.cat((mask_res1_weight_data, mask1_weight_data), 0)
        idbn3.weight.data *= weight_data
        idbn3.bias.data *= weight_data

        return [idconv1, idbn1, nn.ReLU(True), idconv2, idbn2, nn.ReLU(True), idconv3, idbn3, nn.ReLU(True)]


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBNReLU(c1, c_, 1, 1)
        self.cv2 = ConvBNReLU(c1, c_, 1, 1)
        self.cv3 = ConvBNReLU(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBNReLU(c1, c_, 1, 1)
        self.cv2 = ConvBNReLU(c_ * (len(k) + 1), c2, 1, 1)
        num_3x3_maxpool = []
        max_pool_module_list = []
        for pool_kernel in k:
            assert (pool_kernel-3) % 2 == 0;     "Required Kernel size cannot be implemented with kernel_size of 3"
            num_3x3_maxpool = 1 + (pool_kernel-3)//2
            max_pool_module_list.append(nn.Sequential(*num_3x3_maxpool*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]))
            #max_pool_module_list[-1] = nn.ModuleList(max_pool_module_list[-1])
        self.m = nn.ModuleList(max_pool_module_list)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        slice_kernel = 3
        slice_stride = 2
        self.conv_slice = ConvBNReLU(c1, c1*4, slice_kernel, slice_stride, p, g)
        self.conv = ConvBNReLU(c1 * 4, c2, k, s, p, g)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        #return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        #replace slice operations with conv
        x = self.conv_slice(x)
        x = self.conv(x)
        return x
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        """ print("***********************")
        for f in x:
            print(f.shape) """
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, conf=0.25):
        super(NMS, self).__init__()
        self.conf=conf

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class NMS_Export(nn.Module):
    # Non-Maximum Suppression (NMS) module used while exporting ONNX model
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, conf=0.001):
        super(NMS_Export, self).__init__()
        self.conf = conf

    def forward(self, x):
        return non_max_suppression_export(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 85
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid 
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv  

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # print(str(i)+str(x[i].shape))
            bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
            x[i]=x[i].view(bs, self.na, self.no, ny*nx).permute(0, 1, 3, 2).view(bs, self.na, ny, nx, self.no).contiguous()
            # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # print(str(i)+str(x[i].shape))

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                #print("**")
                #print(y.shape) #[1, 3, w, h, 85]
                #print(self.grid[i].shape) #[1, 3, w, h, 2]
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                """print("**")
                print(y.shape)  #[1, 3, w, h, 85]
                print(y.view(bs, -1, self.no).shape) #[1, 3*w*h, 85]"""
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Detect_head(nn.Module):
    stride = None  # strides computed during build
    export_TDA2 = False
    export_opencv = True
    def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
        super(Detect_head, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 85
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m = nn.ModuleList(ConvBN(x, self.no * self.na, 1, 1) for x in ch)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export() and self.export_TDA2:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
            return x
        elif torch.onnx.is_in_onnx_export() and self.export_opencv:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # yolov5-edgeai
                x[i] = torch.sigmoid(x[i])
                x[i] = x[i].view(-1, self.no)
            return torch.cat(x, dim=0)
        else:
            z = []  # inference output
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                # print(str(i)+str(x[i].shape))
                bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
                # x[i] = x[i].view(bs, self.na, self.no, ny*nx).permute(0, 1, 3, 2).view(bs, self.na, ny, nx, self.no).contiguous()   # yolop
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()   # yolov5-edgeai

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    z.append(y.view(bs, -1, self.no))
            out = x if self.training else (torch.cat(z, 1), x)
            if not torch.onnx.is_in_onnx_export():
                return out
            else:
                return out[0]

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Segment_head(nn.Module):
    export_TDA2 = False
    export_opencv = True
    def __init__(self, inp, oup, kernel=3, stride=1, padding=None, group=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Segment_head, self).__init__()
        # self.m = ConvBNReLU(inp, oup, kernel=3, stride=1, padding=None, group=1)
        self.m = ConvBN(inp, oup, kernel, stride, padding, bias=False)

    def forward(self, x):
        x = self.m(x)
        x = torch.sigmoid(x)

        if torch.onnx.is_in_onnx_export() and self.export_TDA2 or self.export_opencv:
            return x
        if not torch.onnx.is_in_onnx_export():
            return x
        else:
            x = torch.argmax(x, dim=1)
            return x



"""class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'Image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f'{n} {self.names[int(c)]}s, '  # add to string
                if show or save:
                    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        # str += '%s %.2f, ' % (names[int(cls)], conf)  # label
                        ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(cls) % 10])  # plot
            if save:
                f = f'results{i}.jpg'
                str += f"saved to '{f}'"
                img.save(f)  # save
            if show:
                img.show(f'Image {i}')  # show
            if pprint:
                print(str)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self):
        self.display(save=True)  # save results

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list"""
