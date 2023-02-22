"""
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
# ------------------------------------------------------------------------------
#  Libraries
# ------------------------------------------------------------------------------
import torch, math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}

class conv_bn(nn.Sequential):
    def __init__(self, inp, oup, stride):
        super(conv_bn, self).__init__(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
        )
        self.out_channels = oup


# ------------------------------------------------------------------------------
#  Depthwise Separable Convolution
# ------------------------------------------------------------------------------
class DWSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, use_relu=True):
        super(DWSConv2d, self).__init__()
        self.use_relu = use_relu
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x


# ------------------------------------------------------------------------------
#  Xception block
# ------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        layers = []
        filters = in_filters
        for i in range(reps):
            use_relu = False
            if i != reps-1:
                use_relu = True
            layers.append(DWSConv2d(filters, out_filters, 3, stride=1, padding=1, bias=False, use_relu=use_relu))
            if i == 0:
                filters = out_filters

        if strides != 1:
            layers.append(nn.MaxPool2d(3, strides, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inp):
        x = self.layers(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


# ------------------------------------------------------------------------------
#  Xception
# ------------------------------------------------------------------------------
class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.features = nn.Sequential(
        conv_bn(3, 16, 2),      # 0
        # conv_bn(32, 64, 2),    # 0
        Block(16, 32, 2, 2),       # 1
        Block(32, 64, 2, 2),      # 2
        Block(64, 128, 2, 2),     # 3

        # Block(128, 128, 3, 1),
        # Block(128, 128, 3, 1),
        # Block(128, 128, 3, 1),
        # Block(128, 128, 3, 1),

        Block(128, 128, 3, 1),
        Block(128, 128, 3, 1),
        Block(128, 128, 3, 1),
        Block(128, 128, 3, 1),   # 11

        Block(128, 256, 2, 2),    #

        DWSConv2d(256, 256, 3, 1, 1),   # 13

        # do relu here
        DWSConv2d(256, 256, 3, 1, 1),   #

        nn.MaxPool2d(2, 2),  # 15  # 16->8
        nn.MaxPool2d(2, 2),  # 16  # 8->4
        nn.MaxPool2d(2, 2),  # 17  # 4->2
        )
        # self.num_classes = num_classes
        #
        # self.conv1 = conv_bn(3, 32, 2)      # 0
        # # self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        # # self.bn2 = nn.BatchNorm2d(64)
        # # do relu here
        #
        # self.block1 = Block(32, 64, 2, 2)       # 1
        # self.block2 = Block(64, 128, 2, 2)      # 2
        # self.block3 = Block(128, 256, 2, 2)     # 3
        #
        # self.block4 = Block(256, 256, 3, 1)
        # self.block5 = Block(256, 256, 3, 1)
        # self.block6 = Block(256, 256, 3, 1)
        # self.block7 = Block(256, 256, 3, 1)
        #
        # self.block8 = Block(256, 256, 3, 1)
        # self.block9 = Block(256, 256, 3, 1)
        # self.block10 = Block(256, 256, 3, 1)
        # self.block11 = Block(256, 256, 3, 1)    # 11
        #
        # self.block12 = Block(256, 512, 2, 2)    #
        #
        # self.conv3 = DWSConv2d(512, 512, 3, 1, 1)   # 13
        #
        # # do relu here
        # self.conv4 = DWSConv2d(512, 512, 3, 1, 1)   #
        #
        # self.conv5 = nn.MaxPool2d(2, 2),  # 15  # 16->8
        # self.conv6 = nn.MaxPool2d(2, 2),  # 16  # 8->4
        # self.conv7 = nn.MaxPool2d(2, 2),  # 17  # 4->2

        self.fc = nn.Linear(512, num_classes)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)   # 0

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# ------------------------------------------------------------------------------
#  Instance
# ------------------------------------------------------------------------------
def xception(pretrained=False, **kwargs):
    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model


if __name__ == "__main__":
    import onnx
    import onnxruntime as ort
    import onnxsim
    import torch
    from torch.nn import init
    import os
    import torch.nn as nn

    model = xception(pretrained=False)
    print(model)
    # model = UNet(backbone="mobilenetv2", num_classes=2, pretrained_backbone=None)
    # print(model)
    from torchsummary import summary

    # model = mobilenetv2_to_mobilenetv1(model)

    model = model.to("cuda")
    summary(model, (3, 256, 256))

    do_simplify = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to('cpu')
    model.eval()
    onnx_path = './xception.onnx'
    # Input
    inputs = torch.randn(1, 3, 256, 256)
    torch.onnx.export(model,
                      inputs,
                      onnx_path,
                      verbose=False,
                      opset_version=11,
                      input_names=['data'],
                      output_names=['seg_out'])
    print('convert', onnx_path, 'to onnx finish!!!')

    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    print(onnx.helper.printable_graph(model_onnx.graph))  # print

    if do_simplify:
        print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
        model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_path)

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

