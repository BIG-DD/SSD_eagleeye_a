import torch
from models.UNet import MCnet_resnet18
import torch.nn as nn
from models.RM_Prune import ConvBNReLU, ConvBN, Bottleneck_RM_prune, ResBlock_RM_prune, get_mode_4_threshold, Conv2d
from models.backbonds.MobileNetV2 import InvertedResidual, InvertedResidualExp,InvertedResidualExp_RM, InvertedResidualRes,InvertedResidualRes_RM, InvertedResidualNoRes, InvertedResidualNoRes_RM, SparseGate


checkpoint = '/media/byd/A264AC9264AC6AAD/DataSet/12_7_data/eps_e1_2/parking_slot/1227_090452/121.pth'
model = MCnet_resnet18()
trained_dict = torch.load(checkpoint, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
model.eval()


for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        weight = m.weight.data
        bias = m.bias
        running_mean = m.running_mean
        running_var = m.running_var
        print(f'weight: max: {torch.max(weight)}, min: {torch.min(weight)}\n')
        print(f'bias: max: {torch.max(bias)}, min: {torch.min(bias)}\n')
        print(f'running_mean: max: {torch.max(running_mean)}, min: {torch.min(running_mean)}\n')
        print(f'running_var: max: {torch.max(running_var)}, min: {torch.min(running_var)}\n')


# for m in model.modules():
#     if isinstance(m, ConvBNReLU):
#         print('ConvBNReLU\n')
#         for mm in m.modules():
#             if isinstance(mm, nn.Conv2d):
#                 data = mm.weight.data
#                 print('max: ', torch.max(data), ' min: ', torch.min(data))
#     if isinstance(m, InvertedResidualExp):
#         print('InvertedResidualExp\n')
#         for mm in m.modules():
#             if isinstance(mm, nn.Conv2d):
#                 data = mm.weight.data
#                 print('max: ', torch.max(data), ' min: ', torch.min(data))
#     if isinstance(m, InvertedResidualNoRes):
#         print('InvertedResidualNoRes\n')
#         for mm in m.modules():
#             if isinstance(mm, nn.Conv2d):
#                 data = mm.weight.data
#                 print('max: ', torch.max(data), ' min: ', torch.min(data))
#     if isinstance(m, InvertedResidualRes):
#         print('InvertedResidualRes\n')
#         for mm in m.modules():
#             if isinstance(mm, nn.Conv2d):
#                 data = mm.weight.data
#                 print('max: ', torch.max(data), ' min: ', torch.min(data))
#     if isinstance(m, Conv2d):
#         print('Conv2d\n')
#         for mm in m.modules():
#             if isinstance(mm, nn.Conv2d):
#                 data = mm.weight.data
#                 print('max: ', torch.max(data), ' min: ', torch.min(data))
#     if isinstance(m, ConvBN):
#         print('ConvBN\n')
#         for mm in m.modules():
#             if isinstance(mm, nn.Conv2d):
#                 data = mm.weight.data
#                 print('max: ', torch.max(data), ' min: ', torch.min(data))
