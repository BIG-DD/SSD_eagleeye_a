import torch
import torch.nn as nn
__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual_shufflenetv2(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual_shufflenetv2, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):#[4, 8, 4], [24, 48, 96, 192, 1024]
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual_shufflenetv2(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual_shufflenetv2(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x


def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)

    return model


def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress, [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)

if __name__ == "__main__":
    # model = MCnet_resnet18(block_cfg=UNet_MobileNetV2)
    # print(model)
    model = shufflenet_v2_x0_5(pretrained=False, progress=True)
    print(model)
    from torchsummary import summary

    # model = mobilenetv2_to_mobilenetv1(model)

    model = model.to("cuda")
    summary(model, (3, 256, 256))



#
# ShuffleNetV2(
#   (conv1): Sequential(
#     (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#   )
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (stage2): Sequential(
#     (0): InvertedResidual(
#       (branch1): Sequential(
#         (0): Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
#         (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (4): ReLU(inplace=True)
#       )
#       (branch2): Sequential(
#         (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
#         (4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (1): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
#         (4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (2): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
#         (4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (3): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
#         (4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#   )
#   (stage3): Sequential(
#     (0): InvertedResidual(
#       (branch1): Sequential(
#         (0): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
#         (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (3): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (4): ReLU(inplace=True)
#       )
#       (branch2): Sequential(
#         (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
#         (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (1): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
#         (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (2): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
#         (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (3): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
#         (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (4): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
#         (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (5): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
#         (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (6): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
#         (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (7): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
#         (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#   )
#   (stage4): Sequential(
#     (0): InvertedResidual(
#       (branch1): Sequential(
#         (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
#         (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (4): ReLU(inplace=True)
#       )
#       (branch2): Sequential(
#         (0): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
#         (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (1): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
#         (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (2): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
#         (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#     (3): InvertedResidual(
#       (branch2): Sequential(
#         (0): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
#         (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (6): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (7): ReLU(inplace=True)
#       )
#     )
#   )
#   (conv5): Sequential(
#     (0): Conv2d(192, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#   )
#   (fc): Linear(in_features=1024, out_features=1000, bias=True)
# )
# ================================================================
# Total params: 1,366,792
# Trainable params: 1,366,792
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 32.73
# Params size (MB): 5.21
# Estimated Total Size (MB): 38.69
# ----------------------------------------------------------------