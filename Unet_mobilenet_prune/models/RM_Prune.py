import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
from models.backbonds.MobileNetV2 import SparseGate, InvertedResidualExp_RM, InvertedResidualNoRes_RM, InvertedResidualRes_RM, InvertedResidualResConcat_RM
from torch.nn import ConvTranspose2d

use_logger = False


# class Concat(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, dimension=1):
#         super(Concat, self).__init__()
#         self.d = dimension
#
#     def forward(self, x1, x2):
#         """ print("***********************")
#         for f in x:
#             print(f.shape) """
#         return torch.cat([x1, x2], dim=self.d)
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


class Conv2d(nn.Module):
    # Standard convolution
    def __init__(self, inp, oup, kernel_size=1, stride=1, padding=0, group=1,
                 bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, groups=group, bias=True)

    def forward(self, x):
        return self.conv(x)


class ConvBN(nn.Module):
    # Standard convolution
    def __init__(self, inp, oup, kernel=3, stride=1, padding=0, group=1,
                 bias=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel, stride, padding, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(oup, eps=1e-1)

    def forward(self, x):
        return self.bn(self.conv(x))
        # return self.conv(x)


class ConvBNReLU(nn.Module):
    # Standard convolution
    def __init__(self, inp, oup, kernel=3, stride=1, padding=0,
                 groups=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(oup, eps=1e-1)
        self.sparse = SparseGate(oup)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.sparse(x)
        x = self.act(x)
        return x
        # return self.act(self.sparse(self.bn(self.conv(x))))
        # return self.act(self.sparse(self.conv(x)))


#  add mask
class ConvBNReLU_mask(nn.Module):
    # Standard convolution
    def __init__(self, inp, oup, kernel=3, stride=1, padding=0,
                 group=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvBNReLU_mask, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel, stride, padding, groups=group, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.mask = nn.Conv2d(oup, oup, 1, groups=oup, bias=False)
        nn.init.ones_(self.mask.weight)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.mask(self.bn(self.conv(x))))


class ConvBN_mask(nn.Module):
    # Standard convolution
    def __init__(self, inp, oup, kernel=3, stride=1, padding=0, group=1,
                 bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvBN_mask, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel, stride, padding, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(oup)
        self.mask = nn.Conv2d(oup, oup, 1, groups=oup, bias=False)
        nn.init.ones_(self.mask.weight)

    def forward(self, x):
        return self.mask(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        mid_planes = 2 * out_planes
        assert mid_planes >= in_planes
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


class Bottleneck_mask_RM(nn.Module):
    # Standard bottleneck
    def __init__(self, in_planes, out_planes, shortcut=True,
                 expand_ratio=0.5):  # ch_in, ch_out, shortcut, groups, expansion=0.5
        super(Bottleneck_mask_RM, self).__init__()
        self.add = shortcut and in_planes == out_planes
        self.in_planes = in_planes
        self.mid_planes = int(out_planes * expand_ratio)  # hidden channels
        self.out_planes = out_planes

        self.mask_res1 = nn.Sequential(
            *[nn.Conv2d(self.in_planes, self.in_planes, 1, groups=self.in_planes, bias=False),
              nn.ReLU(inplace=True)])
        self.mask_res2 = nn.Sequential(
            *[nn.Conv2d(self.in_planes, self.in_planes, 1, groups=self.in_planes, bias=False),
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
        idconv1 = nn.Conv2d(self.in_planes, self.in_planes + self.mid_planes, kernel_size=1, stride=1, padding=0,
                            bias=False).eval()
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
        idconv2 = nn.Conv2d(self.in_planes + self.mid_planes, self.in_planes + self.out_planes, kernel_size=3, stride=1,
                            padding=1, bias=False).eval()
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
        idconv2.weight.data[self.in_planes:, self.in_planes:] = self.conv2.weight.data  # , self.in_planes:
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
        idconv3 = nn.Conv2d(self.in_planes + self.out_planes, self.out_planes, kernel_size=1, stride=1, padding=0,
                            bias=False).eval()
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


# RM：模型结构进行去残差连接
def merge_mask(model):
    def foo(net):
        global blocks
        childrens = list(net.children())
        if isinstance(net, ResBlock) or isinstance(net, Bottleneck_mask_RM):
            blocks += net.deploy()

        elif not childrens:
            if isinstance(net, nn.Conv2d) and net.groups != 1:
                # print('start')
                # print(blocks[-1].weight.data)
                # print(blocks[-1].bias.data)
                blocks[-1].weight.data *= net.weight.data.reshape(-1)
                blocks[-1].bias.data *= net.weight.data.reshape(-1)
                # print(blocks[-1].weight.data)
                # print(blocks[-1].bias.data)
                # print('end')
            else:
                blocks += [net]
        else:
            for c in childrens:
                foo(c)

    global blocks

    blocks = []
    foo(model.eval())
    return nn.Sequential(*blocks)


# prune后卷积核能被4整除
def get_mode_4_mask(m, threshold):
    # bn = torch.zeros(m.weight.data.abs().shape[0])
    # bn[:] = m.weight.data.abs().clone()
    # y, index = torch.sort(bn)  # descending=True 从大到小排序，descending=False 从小到大排序，默认 descending=Flase
    # threshold = y[int(len(bn)*0.5)]

    mask = m.weight.data.abs().reshape(-1) > threshold
    print('mask:' + str(len(mask)) + '   mask>0:' + str(mask.sum()))
    # # print(int(mask.sum()))
    # if int(mask.sum()) % 4 != 0:
    #     mask_num = int(mask.sum()) + 4 - (int(mask.sum()) % 4)
    #     bn = torch.zeros(m.weight.data.shape[0])
    #     bn[:] = m.weight.data.abs().clone()
    #     val, index = torch.sort(bn, descending=True)
    #     for j in range(mask_num):
    #         ind = index[j].type(torch.long)
    #         mask[ind:ind+1] = True
    #     # print(int(mask.sum()))
    return mask

def get_spars_out_mask(m, threshold):
    mask = m.weight.data.abs().reshape(-1) >= threshold
    print('mask:' + str(len(mask)) + '   mask>0:' + str(mask.sum()))
    return mask

def get_spars_out_mask_channel(m, channel):
    out_channel = m.out_channels
    mask = torch.zeros(out_channel, dtype=torch.bool)
    weight_sort, indx = torch.sort(m.weight.data.abs().reshape(-1), descending=True)
    get_indx = indx[:channel]
    mask[get_indx] = True
    return mask


# 计算新的threshold，卷积核能被4整除，且卷积核权重为0的数量不小于8
def get_mode_4_threshold(m, threshold):
    mask = m.weight.data.abs().reshape(-1) > threshold
    # print('mask:' + str(len(mask)) + '   mask>0:' + str(int(mask.sum())))
    if 8 >= len(mask):  # 卷积核数量小于等于8，将不修改权重
        re_threshold = 0
    else:
        mask_num = int(mask.sum())
        if int(mask.sum()) % 4 != 0:    # 计算卷积核能被4整除的数量
            mask_num = int(mask.sum()) - (int(mask.sum()) % 4)
        if mask_num >= len(mask):   # 阈值不需要修改
            re_threshold = threshold
        else:
            if 8 > mask_num:    #
                mask_num = 8
            bn = torch.zeros(len(mask))
            bn[:] = m.weight.data.abs().reshape(-1).clone()
            val, index = torch.sort(bn, descending=True)
            re_threshold = val[mask_num]
    # mask = m.weight.data.abs().reshape(-1) > re_threshold
    # print('mask:' + str(len(mask)) + '   mask>0:' + str(int(mask.sum())))
    return re_threshold


# 获得每一层卷积操作的输入输出大小
def get_block_params(features, in_mask, threshold):
    in_mask = int(in_mask.sum())
    layer_params = []  # kernel, stride, padding, groups
    for i, m in enumerate(features):
        if isinstance(m, nn.BatchNorm2d):
            mask = get_mode_4_mask(m, threshold)
            out_mask = mask  # 下一层卷积操作的输入
            mask = int(mask.sum())
            kernel_size = features[i - 1].kernel_size
            stride = features[i - 1].stride
            padding = features[i - 1].padding
            groups = features[i - 1].groups
            layer_params.append([in_mask, mask, kernel_size, stride, padding, groups])
            in_mask = mask

    return layer_params, out_mask


# 只适用于RM后为conv+bn+relu的模块
class ResBlock_RM_prune(nn.Module):
    # Standard bottleneck
    def __init__(self, layer_params):  # layer_params = []    # in_mask, out_mask, kernel_size, stride, padding, groups
        super(ResBlock_RM_prune, self).__init__()
        layers = []
        for layer_param in layer_params:
            layers.append(nn.Conv2d(layer_param[0], layer_param[1], kernel_size=layer_param[2],
                                    stride=layer_param[3], padding=layer_param[4], groups=layer_param[5],
                                    bias=False))  #
            layers.append(nn.BatchNorm2d(layer_param[1]))
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # print(x)
        out = self.block(x)
        # print(out)
        return out


# RM+prune, 只适用于RM后为conv+bn+relu的模块，后对新的模块进行赋值
def block2block_RM_prune(layers, in_mask, threshold, concat_num, block_out_mask, logger):
    if use_logger:
        logger.info('block2block_RM_prune')
    if isinstance(layers, ResBlock):
        features = layers.deploy()
    else:
        features = merge_mask(copy.deepcopy(layers))

    conv_size = features[0].weight.data.shape
    if conv_size[1] != len(in_mask):
        # print(int(block_out_mask[concat_num[-1]].sum()))
        in_mask = torch.cat((in_mask, block_out_mask[concat_num[-1]]), 0)
    block_params, out_mask = get_block_params(features, in_mask, threshold)
    RM_prune_layers = ResBlock_RM_prune(block_params)

    for i, m in enumerate(features):
        if isinstance(m, nn.BatchNorm2d):
            if use_logger:
                logger.info(int(in_mask.sum()))
                logger.info(in_mask)
            mask = get_mode_4_mask(m, threshold)

            RM_prune_layers.block[i - 1].weight.data = features[i - 1].weight.data[mask][:, in_mask]
            RM_prune_layers.block[i].weight.data = m.weight.data[mask]
            RM_prune_layers.block[i].bias.data = m.bias.data[mask]
            RM_prune_layers.block[i].running_mean = m.running_mean[mask]
            RM_prune_layers.block[i].running_var = m.running_var[mask]
            in_mask = mask

    return RM_prune_layers, out_mask, [block_params]


# RM+prune, 只适用于RM后为Bottleneck的模块
class Bottleneck_RM_prune(nn.Module):
    # Standard bottleneck
    def __init__(self, layer_params):  # layer_params = []    # in_mask, out_mask, kernel_size, stride, padding, groups
        super(Bottleneck_RM_prune, self).__init__()
        self.idconv1 = nn.Conv2d(layer_params[0][0], layer_params[0][1], kernel_size=layer_params[0][2],
                                 stride=layer_params[0][3], padding=layer_params[0][4], groups=layer_params[0][5],
                                 bias=False)
        self.idbn1 = nn.BatchNorm2d(layer_params[0][1])
        self.idrelu1 = nn.ReLU(inplace=True)

        self.idconv2 = nn.Conv2d(layer_params[1][0], layer_params[1][1], kernel_size=layer_params[1][2],
                                 stride=layer_params[1][3], padding=layer_params[1][4], groups=layer_params[1][5],
                                 bias=False)
        self.idbn2 = nn.BatchNorm2d(layer_params[1][1])
        self.idrelu2 = nn.ReLU(inplace=True)

        self.idconv3 = nn.Conv2d(layer_params[2][0], layer_params[2][1], kernel_size=layer_params[2][2],
                                 stride=layer_params[2][3], padding=layer_params[2][4], groups=layer_params[2][5],
                                 bias=False)
        self.idbn3 = nn.BatchNorm2d(layer_params[2][1])
        self.idrelu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x)
        out = self.idrelu1(self.idbn1(self.idconv1(x)))
        # print(out)
        out = self.idrelu2(self.idbn2(self.idconv2(out)))
        # print(out)
        out = self.idrelu3(self.idbn3(self.idconv3(out)))
        # print(out)
        return out


# RM+prune后对新的模块进行赋值
def Bottleneck2Bottleneck_RM_prune(Bottleneck_layer, in_mask, threshold, logger):
    if use_logger:
        logger.info('Bottleneck2Bottleneck_RM_prune\n')
    if isinstance(Bottleneck_layer, Bottleneck_mask_RM):
        features = Bottleneck_layer.deploy()
    else:
        features = merge_mask(copy.deepcopy(Bottleneck_layer))

    layer_params, out_mask = get_block_params(features, in_mask, threshold)

    RM_prune_layer = Bottleneck_RM_prune(layer_params)
    count = 0
    for i, m in enumerate(features):
        if isinstance(m, nn.BatchNorm2d):
            if use_logger:
                logger.info(int(in_mask.sum()))
                logger.info(in_mask)
            count += 1
            mask = get_mode_4_mask(m, threshold)
            if count == 1:
                RM_prune_layer.idconv1.weight.data = features[i - 1].weight.data[mask][:, in_mask]
                RM_prune_layer.idbn1.weight.data = m.weight.data[mask]
                RM_prune_layer.idbn1.bias.data = m.bias.data[mask]
                RM_prune_layer.idbn1.running_mean = m.running_mean[mask]
                RM_prune_layer.idbn1.running_var = m.running_var[mask]
            elif count == 2:
                RM_prune_layer.idconv2.weight.data = features[i - 1].weight.data[mask][:, in_mask]
                RM_prune_layer.idbn2.weight.data = m.weight.data[mask]
                RM_prune_layer.idbn2.bias.data = m.bias.data[mask]
                RM_prune_layer.idbn2.running_mean = m.running_mean[mask]
                RM_prune_layer.idbn2.running_var = m.running_var[mask]
            elif count == 3:
                RM_prune_layer.idconv3.weight.data = features[i - 1].weight.data[mask][:, in_mask]
                RM_prune_layer.idbn3.weight.data = m.weight.data[mask]
                RM_prune_layer.idbn3.bias.data = m.bias.data[mask]
                RM_prune_layer.idbn3.running_mean = m.running_mean[mask]
                RM_prune_layer.idbn3.running_var = m.running_var[mask]

            in_mask = mask

    return RM_prune_layer, out_mask, [layer_params]


def get_prune_threshold(model, prune_scale_rate=0.1):
    layers = merge_mask(copy.deepcopy(model))
    # prune scale rate (default:0.5)
    total = 0
    for m in layers:
        # print(m)
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    pos = 0
    for m in layers:
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[pos:(pos + size)] = m.weight.data.abs().clone()
            pos += size

    y, index = torch.sort(bn)  # descending=True 从大到小排序，descending=False 从小到大排序，默认 descending=Flase
    thre_index = int(total * prune_scale_rate)
    threshold = y[thre_index]
    if prune_scale_rate == 0.0:
        for i, val in enumerate(y):
            if val > prune_scale_rate:
                print(i)
                threshold = prune_scale_rate  # val-0.0000000000001
                break
    return threshold


# RM+prune, 只适用于RM后为ConvBNReLU的模块，后对新的模块进行赋值
def ConvBNReLU2ConvBNReLU_RM_prune(block, in_mask, threshold, concat_num, block_out_mask, logger):
    features = merge_mask(copy.deepcopy(block))
    for i, m in enumerate(features):
        if isinstance(m, nn.BatchNorm2d):
            kernel_size = features[i - 1].kernel_size
            stride = features[i - 1].stride
            padding = features[i - 1].padding
            groups = features[i - 1].groups
            out_mask = get_mode_4_mask(m, threshold)

    conv_size = features[0].weight.data.shape  # [out, in, kernel, kernel]
    if conv_size[1] != len(in_mask):
        in_mask = torch.cat((in_mask, block_out_mask[concat_num[-1]]), 0)
    if use_logger:
        logger.info('ConvBNReLU2ConvBNReLU_RM_prune')
        logger.info(int(in_mask.sum()))
        logger.info(in_mask)
    # print('out_mask:' + str(len(out_mask)) + '   out_mask>0:' + str(out_mask.sum()))

    prune_layer = ConvBNReLU(int(in_mask.sum()), int(out_mask.sum()), kernel=kernel_size, stride=stride,
                             padding=padding, groups=groups)
    prune_layer.conv.weight.data = features[0].weight.data[out_mask][:, in_mask]
    prune_layer.bn.weight.data = features[1].weight.data[out_mask]
    prune_layer.bn.bias.data = features[1].bias.data[out_mask]
    prune_layer.bn.running_mean = features[1].running_mean[out_mask]
    prune_layer.bn.running_var = features[1].running_var[out_mask]
    return prune_layer, out_mask, [int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, groups]


# RM+prune, 只适用于RM后为ConvBN的模块，后对新的模块进行赋值
def ConvBN2ConvBN_RM_prune(block, in_mask, threshold, concat_num, block_out_mask, logger):
    features = merge_mask(copy.deepcopy(block))
    for i, m in enumerate(features):
        if isinstance(m, nn.BatchNorm2d):
            kernel_size = features[i - 1].kernel_size
            stride = features[i - 1].stride
            padding = features[i - 1].padding
            groups = features[i - 1].groups
            out_mask = get_mode_4_mask(m, threshold)

    conv_size = features[0].weight.data.shape  # [out, in, kernel, kernel]
    if conv_size[1] != len(in_mask):
        in_mask = torch.cat((in_mask, block_out_mask[concat_num[-1]]), 0)
    if use_logger:
        logger.info('ConvBN2ConvBN_RM_prune')
        logger.info(int(in_mask.sum()))
        logger.info(in_mask)

    prune_layer = ConvBN(int(in_mask.sum()), int(out_mask.sum()), kernel=kernel_size, stride=stride, padding=padding,
                         group=groups)
    prune_layer.conv.weight.data = features[0].weight.data[out_mask][:, in_mask]
    prune_layer.bn.weight.data = features[1].weight.data[out_mask]
    prune_layer.bn.bias.data = features[1].bias.data[out_mask]
    prune_layer.bn.running_mean = features[1].running_mean[out_mask]
    prune_layer.bn.running_var = features[1].running_var[out_mask]
    return prune_layer, out_mask, [int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, groups]


# RM+prune, 只适用于RM后为Conv的模块，后对新的模块进行赋值
def Conv2Conv_prune(block, in_mask):
    features = merge_mask(copy.deepcopy(block))
    kernel_size = features[0].kernel_size
    stride = features[0].stride
    padding = features[0].padding
    groups = features[0].groups

    conv_size = features[0].weight.data.shape  # [out, in, kernel, kernel]
    out_mask = torch.ones(conv_size[0])
    out_mask = out_mask > 0

    prune_layer = Conv2d(int(in_mask.sum()), int(out_mask.sum()), kernel_size=kernel_size, stride=stride,
                         padding=padding, group=groups)
    prune_layer.conv.weight.data = features[0].weight.data[out_mask][:, in_mask]
    prune_layer.conv.bias.data = features[0].bias.data
    return prune_layer, out_mask, [int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, groups]

def ConvBN_prune(block, in_mask):
    features = merge_mask(copy.deepcopy(block))
    kernel_size = features[0].kernel_size
    stride = features[0].stride
    padding = features[0].padding
    groups = features[0].groups

    conv_size = features[0].weight.data.shape  # [out, in, kernel, kernel]
    out_mask = torch.ones(conv_size[0])
    out_mask = out_mask > 0

    prune_layer = ConvBN(int(in_mask.sum()), int(out_mask.sum()), kernel=kernel_size, stride=stride,
                         padding=padding, group=groups)
    prune_layer.conv.weight.data = features[0].weight.data[out_mask][:, in_mask]
    prune_layer.bn.weight.data = features[1].weight.data[out_mask]
    prune_layer.bn.bias.data = features[1].bias.data[out_mask]
    prune_layer.bn.running_mean.data = features[1].running_mean.data[out_mask]
    prune_layer.bn.running_var.data = features[1].running_var.data[out_mask]
    return prune_layer, out_mask, [int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, groups]

def ConvBNReLU_prune(block, in_mask):

    prune_layer = copy.deepcopy(block)
    kernel_size = prune_layer.conv.kernel_size
    stride = prune_layer.conv.stride
    padding = prune_layer.conv.padding
    conv_size = prune_layer.conv.weight.data.shape  # [out, in, kernel, kernel]
    out_mask = torch.ones(conv_size[0], dtype=torch.bool)
    return block, out_mask, [int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding]

def InvertedResidualExp_prune(block, in_mask, thr):
    if isinstance(block.conv[5], SparseGate):
        m = block.conv[5]._conv
        out_mask = get_spars_out_mask(m, thr[0])
    dw_inp = int(in_mask.sum())
    dw_oup = block.conv[0].out_channels
    dw_stride = block.conv[0].stride[0]
    pw_inp = block.conv[3].in_channels
    pw_oup = int(out_mask.sum())
    prune_layer = InvertedResidualExp_RM(dw_inp, dw_oup, dw_stride, pw_inp, pw_oup)
    prune_layer.conv[0].weight.data = block.conv[0].weight.data
    prune_layer.conv[1].weight.data = block.conv[1].weight.data
    prune_layer.conv[1].bias.data = block.conv[1].bias.data
    prune_layer.conv[1].running_mean.data = block.conv[1].running_mean.data
    prune_layer.conv[1].running_var.data = block.conv[1].running_var.data
    prune_layer.conv[3].weight.data = block.conv[3].weight.data[out_mask, :, :, :]
    prune_layer.conv[4].weight.data = block.conv[4].weight.data[out_mask]*block.conv[5]._conv.weight.data.reshape(-1)[out_mask]
    prune_layer.conv[4].bias.data = block.conv[4].bias.data[out_mask]*block.conv[5]._conv.weight.data.reshape(-1)[out_mask]
    prune_layer.conv[4].running_mean.data = block.conv[4].running_mean.data[out_mask]
    prune_layer.conv[4].running_var.data = block.conv[4].running_var.data[out_mask]

    return prune_layer, out_mask, [dw_inp, dw_oup, dw_stride, pw_inp, pw_oup]


def InvertedResidualNoRes_prune(block, in_mask, thr, concat_num, block_out_mask):

    in_channels = block.conv[0].in_channels  # [out, in, kernel, kernel]
    if in_channels != len(in_mask):
        in_mask = torch.cat((in_mask, block_out_mask[concat_num[-1]]), 0)
    out_mask1 = get_spars_out_mask(block.conv[2]._conv, thr[0])
    # out_mask2 = get_spars_out_mask(block.conv[9]._conv, thr[1])
    out_mask2 = torch.ones(block.conv[9]._conv.out_channels, dtype=torch.bool)
    pw_inp = int(in_mask.sum())
    pw_oup = int(out_mask1.sum())
    dw_inp = pw_oup
    dw_oup = pw_oup
    dw_stride = block.conv[4].stride[0]
    pwl_inp = pw_oup
    pwl_oup = int(out_mask2.sum())
    prune_layer = InvertedResidualNoRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    prune_layer.conv[0].weight.data = block.conv[0].weight.data[:, in_mask, :, :][out_mask1, :, :, :]
    prune_layer.conv[1].weight.data = block.conv[1].weight.data[out_mask1]*block.conv[2]._conv.weight.data.reshape(-1)[out_mask1]
    prune_layer.conv[1].bias.data = block.conv[1].bias.data[out_mask1]*block.conv[2]._conv.weight.data.reshape(-1)[out_mask1]
    prune_layer.conv[1].running_mean.data = block.conv[1].running_mean.data[out_mask1]
    prune_layer.conv[1].running_var.data = block.conv[1].running_var.data[out_mask1]
    prune_layer.conv[3].weight.data = block.conv[4].weight.data[out_mask1, :, :, :]
    prune_layer.conv[4].weight.data = block.conv[5].weight.data[out_mask1]
    prune_layer.conv[4].bias.data = block.conv[5].bias.data[out_mask1]
    prune_layer.conv[4].running_mean.data = block.conv[5].running_mean.data[out_mask1]
    prune_layer.conv[4].running_var.data = block.conv[5].running_var.data[out_mask1]
    prune_layer.conv[6].weight.data = block.conv[7].weight.data[:, out_mask1, :, :][out_mask2, :, :, :]
    prune_layer.conv[7].weight.data = block.conv[8].weight.data[out_mask2]*block.conv[9]._conv.weight.data.reshape(-1)[out_mask2]
    prune_layer.conv[7].bias.data = block.conv[8].bias.data[out_mask2]*block.conv[9]._conv.weight.data.reshape(-1)[out_mask2]
    prune_layer.conv[7].running_mean.data = block.conv[8].running_mean.data[out_mask2]
    prune_layer.conv[7].running_var.data = block.conv[8].running_var.data[out_mask2]
    return prune_layer, out_mask2, [pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup]

def InvertedResidualRes_prune(block, in_mask, thr):
    out_mask1 = get_spars_out_mask(block.conv[2]._conv, thr[0])
    pw_inp = int(in_mask.sum())
    pw_oup = int(out_mask1.sum())
    dw_inp = pw_oup
    dw_oup = pw_oup
    dw_stride = block.conv[4].stride[0]
    pwl_inp = pw_oup
    pwl_oup = block.conv[7].out_channels
    out_mask2 = torch.ones(pw_inp, dtype=torch.bool)
    if pw_inp != pwl_oup:
        out_mask2 = get_spars_out_mask_channel(block.conv[9]._conv, pw_inp)
        pwl_oup = int(out_mask2.sum())
    out_mask = in_mask
    prune_layer = InvertedResidualRes_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup)
    prune_layer.conv[0].weight.data = block.conv[0].weight.data[:, in_mask, :, :][out_mask1, :, :, :]
    prune_layer.conv[1].weight.data = block.conv[1].weight.data[out_mask1]*block.conv[2]._conv.weight.data.reshape(-1)[out_mask1]
    prune_layer.conv[1].bias.data = block.conv[1].bias.data[out_mask1]*block.conv[2]._conv.weight.data.reshape(-1)[out_mask1]
    prune_layer.conv[1].running_mean.data = block.conv[1].running_mean.data[out_mask1]
    prune_layer.conv[1].running_var.data = block.conv[1].running_var.data[out_mask1]
    prune_layer.conv[3].weight.data = block.conv[4].weight.data[out_mask1, :, :, :]
    prune_layer.conv[4].weight.data = block.conv[5].weight.data[out_mask1]
    prune_layer.conv[4].bias.data = block.conv[5].bias.data[out_mask1]
    prune_layer.conv[4].running_mean.data = block.conv[5].running_mean.data[out_mask1]
    prune_layer.conv[4].running_var.data = block.conv[5].running_var.data[out_mask1]
    prune_layer.conv[6].weight.data = block.conv[7].weight.data[:, out_mask1, :, :][out_mask2, :, :, :]
    prune_layer.conv[7].weight.data = block.conv[8].weight.data[out_mask2]*block.conv[9]._conv.weight.data.reshape(-1)[out_mask2]
    prune_layer.conv[7].bias.data = block.conv[8].bias.data[out_mask2]*block.conv[9]._conv.weight.data.reshape(-1)[out_mask2]
    prune_layer.conv[7].running_mean.data = block.conv[8].running_mean.data[out_mask2]
    prune_layer.conv[7].running_var.data = block.conv[8].running_var.data[out_mask2]
    return prune_layer, out_mask, [pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup]

def InvertedResidualResConcat_prune(block, in_mask, thr):
    out_mask1 = get_spars_out_mask(block.conv[2]._conv, thr[0])
    out_mask2 = get_spars_out_mask(block.conv[9]._conv, thr[1])
    out_mask3 = get_spars_out_mask(block.conv1_1[2]._conv, thr[2])
    pw_inp = int(in_mask.sum())
    pw_oup = int(out_mask1.sum())
    dw_inp = pw_oup
    dw_oup = pw_oup
    dw_stride = block.conv[4].stride[0]
    pwl_inp = pw_oup
    pwl_oup = int(out_mask2.sum())
    cat_inp = int(in_mask.sum() + out_mask2.sum())
    cat_oup = int(out_mask3.sum())
    cat_inp_mask = torch.cat((in_mask, out_mask2), dim=0)
    out_mask = out_mask3
    prune_layer = InvertedResidualResConcat_RM(pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup, cat_inp, cat_oup)
    prune_layer.conv[0].weight.data = block.conv[0].weight.data[:, in_mask, :, :][out_mask1, :, :, :]
    prune_layer.conv[1].weight.data = block.conv[1].weight.data[out_mask1]*block.conv[2]._conv.weight.data.reshape(-1)[out_mask1]
    prune_layer.conv[1].bias.data = block.conv[1].bias.data[out_mask1]*block.conv[2]._conv.weight.data.reshape(-1)[out_mask1]
    prune_layer.conv[1].running_mean.data = block.conv[1].running_mean.data[out_mask1]
    prune_layer.conv[1].running_var.data = block.conv[1].running_var.data[out_mask1]
    prune_layer.conv[3].weight.data = block.conv[4].weight.data[out_mask1, :, :, :]
    prune_layer.conv[4].weight.data = block.conv[5].weight.data[out_mask1]
    prune_layer.conv[4].bias.data = block.conv[5].bias.data[out_mask1]
    prune_layer.conv[4].running_mean.data = block.conv[5].running_mean.data[out_mask1]
    prune_layer.conv[4].running_var.data = block.conv[5].running_var.data[out_mask1]
    prune_layer.conv[6].weight.data = block.conv[7].weight.data[:, out_mask1, :, :][out_mask2, :, :, :]
    prune_layer.conv[7].weight.data = block.conv[8].weight.data[out_mask2]*block.conv[9]._conv.weight.data.reshape(-1)[out_mask2]
    prune_layer.conv[7].bias.data = block.conv[8].bias.data[out_mask2]*block.conv[9]._conv.weight.data.reshape(-1)[out_mask2]
    prune_layer.conv[7].running_mean.data = block.conv[8].running_mean.data[out_mask2]
    prune_layer.conv[7].running_var.data = block.conv[8].running_var.data[out_mask2]
    prune_layer.conv1_1[0].weight.data = block.conv1_1[0].weight.data[:, cat_inp_mask, :, :][out_mask3, :, :, :]
    prune_layer.conv1_1[1].weight.data = block.conv1_1[1].weight.data[out_mask3]*block.conv1_1[2]._conv.weight.data.reshape(-1)[out_mask3]
    prune_layer.conv1_1[1].bias.data = block.conv1_1[1].bias.data[out_mask3]*block.conv1_1[2]._conv.weight.data.reshape(-1)[out_mask3]
    prune_layer.conv1_1[1].running_mean.data = block.conv1_1[1].running_mean.data[out_mask3]
    prune_layer.conv1_1[1].running_var.data = block.conv1_1[1].running_var.data[out_mask3]
    return prune_layer, out_mask, [pw_inp, pw_oup, dw_inp, dw_oup, dw_stride, pwl_inp, pwl_oup, cat_inp, cat_oup]

def ConvTranspose2d_prune(block, in_mask):
    out_mask = in_mask

    kernel_size = block.kernel_size
    stride = block.stride
    padding = block.padding
    groups = int(out_mask.sum())

    prune_layer = ConvTranspose2d(int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, 0, groups, False)
    # print(prune_layer.weight.data)
    prune_layer.weight.data = block.weight.data[out_mask]
    return prune_layer, out_mask, [int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, 0, groups, False]

# mobilenetv2->mobilenetv1

def fuse_cbcb(conv1, bn1, conv2, bn2):
    inp=conv1.in_channels
    mid=conv1.out_channels
    oup=conv2.out_channels
    conv1=torch.nn.utils.fuse_conv_bn_eval(conv1.eval(),bn1.eval())
    fused_conv=nn.Conv2d(inp,oup,1,bias=False)
    fused_conv.weight.data=(conv2.weight.data.view(oup,mid)@conv1.weight.data.view(mid,-1)).view(oup,inp,1,1)
    bn2.running_mean-=conv2.weight.data.view(oup,mid)@conv1.bias.data
    return fused_conv, bn2


def fuse_cb(conv_w, bn_rm, bn_rv, bn_w, bn_b, eps):
    bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = bn_rm * bn_var_rsqrt * bn_w-bn_b
    return conv_w,conv_b


def rm_r(model):
    inp = model.conv[0].in_channels
    mid = inp+model.conv[0].out_channels
    oup = model.conv[6].out_channels

    running1 = nn.BatchNorm2d(inp, affine=False)
    running2 = nn.BatchNorm2d(oup, affine=False)

    idconv1 = nn.Conv2d(inp, mid, kernel_size=1, bias=False).eval()
    idbn1 = nn.BatchNorm2d(mid).eval()

    nn.init.dirac_(idconv1.weight.data[:inp])
    bn_var_sqrt = torch.sqrt(running1.running_var+running1.eps)
    idbn1.weight.data[:inp] = bn_var_sqrt
    idbn1.bias.data[:inp] = running1.running_mean
    idbn1.running_mean.data[:inp] = running1.running_mean
    idbn1.running_var.data[:inp] = running1.running_var

    idconv1.weight.data[inp:] = model.conv[0].weight.data
    idbn1.weight.data[inp:] = model.conv[1].weight.data
    idbn1.bias.data[inp:] = model.conv[1].bias.data
    idbn1.running_mean.data[inp:] = model.conv[1].running_mean
    idbn1.running_var.data[inp:] = model.conv[1].running_var
    idrelu1 = nn.PReLU(mid)
    torch.nn.init.ones_(idrelu1.weight.data[:inp])
    torch.nn.init.zeros_(idrelu1.weight.data[inp:])

    idconv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=model.stride, padding=1, groups=mid, bias=False).eval()
    idbn2 = nn.BatchNorm2d(mid).eval()

    nn.init.dirac_(idconv2.weight.data[:inp], groups=inp)
    idbn2.weight.data[:inp] = idbn1.weight.data[:inp]
    idbn2.bias.data[:inp] = idbn1.bias.data[:inp]
    idbn2.running_mean.data[:inp] = idbn1.running_mean.data[:inp]
    idbn2.running_var.data[:inp] = idbn1.running_var.data[:inp]

    idconv2.weight.data[inp:] = model.conv[3].weight.data
    idbn2.weight.data[inp:] = model.conv[4].weight.data
    idbn2.bias.data[inp:] = model.conv[4].bias.data
    idbn2.running_mean.data[inp:] = model.conv[4].running_mean
    idbn2.running_var.data[inp:] = model.conv[4].running_var
    idrelu2 = nn.PReLU(mid)
    torch.nn.init.ones_(idrelu2.weight.data[:inp])
    torch.nn.init.zeros_(idrelu2.weight.data[inp:])

    idconv3 = nn.Conv2d(mid, oup, kernel_size=1, bias=False).eval()
    idbn3 = nn.BatchNorm2d(oup).eval()

    nn.init.dirac_(idconv3.weight.data[:, :inp])
    idconv3.weight.data[:, inp:], bias = fuse_cb(model.conv[6].weight, model.conv[7].running_mean, model.conv[7].running_var,
                                                 model.conv[7].weight, model.conv[7].bias, model.conv[7].eps)
    bn_var_sqrt = torch.sqrt(running2.running_var+running2.eps)
    idbn3.weight.data = bn_var_sqrt
    idbn3.bias.data = running2.running_mean
    idbn3.running_mean.data = running2.running_mean+bias
    idbn3.running_var.data = running2.running_var
    return [idconv1, idbn1, idrelu1, idconv2, idbn2, idrelu2, idconv3, idbn3]


class InvertedResidual_2_3(nn.Module):
    def __init__(self):
        super(InvertedResidual_2_3, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 168, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(168),
            nn.PReLU(168),

            nn.Conv2d(168, 168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=168, bias=False),
            nn.BatchNorm2d(168),
            nn.PReLU(168),

            nn.Conv2d(168, 24, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(24)
        )

    def forward(self, x):
        return self.block(x)



class InvertedResidual_4_5_6(nn.Module):
    def __init__(self):
        super(InvertedResidual_4_5_6, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False),
            nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 224, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=224),
            nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=224, bias=False),
            nn.BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=224),
            nn.Conv2d(224, 224, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=224),
            nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=224, bias=False),
            nn.BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=224),
            nn.Conv2d(224, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return self.block(x)


