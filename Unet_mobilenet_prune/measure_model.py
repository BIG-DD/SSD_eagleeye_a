#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import models

import argparse
from time import time

import torch
from torchsummary import summary


#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Arguments for the script")

parser.add_argument('--use_cuda', action='store_true', default=False,
					help='Use GPU acceleration')

parser.add_argument('--input_sz', type=int, default=384,
					help='Size of the input')

parser.add_argument('--n_measures', type=int, default=10,
					help='Number of time measurements')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Create model
#------------------------------------------------------------------------------
# UNet
# model = models.UNet(
# 	backbone="mobilenetv2",
# 	num_classes=2,
# )
model = models.JSEGNET21V2(
	backbone="mobilenetv2",
	num_classes=2,
)

# # DeepLabV3+
# model = DeepLabV3Plus(
#     backbone='resnet18',
#     output_stride=16,
#     num_classes=2,
#     pretrained_backbone=None,
# )

# # BiSeNet
# model = BiSeNet(
#     backbone='resnet18',
#     num_classes=2,
#     pretrained_backbone=None,
# )

# # PSPNet
# model = PSPNet(
# 	backbone='resnet18',
# 	num_classes=2,
# 	pretrained_backbone=None,
# )

# # ICNet
# model = ICNet(
#     backbone='resnet18',
#     num_classes=2,
#     pretrained_backbone=None,
# )


# #------------------------------------------------------------------------------
# #   Summary network
# #------------------------------------------------------------------------------
model.train()
model.summary(input_shape=(3, args.input_sz, args.input_sz), device='cpu')


#------------------------------------------------------------------------------
#   Measure time
#------------------------------------------------------------------------------
input = torch.randn([1, 3, args.input_sz, args.input_sz], dtype=torch.float)
if args.use_cuda:
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	model.cuda()
	input = input.cuda()

for _ in range(10):
	model(input)

start_time = time()
for _ in range(args.n_measures):
	model(input)
finish_time = time()

if args.use_cuda:
	print("Inference time on cuda: %.2f [ms]" % ((finish_time-start_time)*1000/args.n_measures))
	print("Inference fps on cuda: %.2f [fps]" % (1 / ((finish_time-start_time)/args.n_measures)))
else:
	print("Inference time on cpu: %.2f [ms]" % ((finish_time-start_time)*1000/args.n_measures))
	print("Inference fps on cpu: %.2f [fps]" % (1 / ((finish_time-start_time)/args.n_measures)))



# UNet
# Total params: 4,683,331
# Trainable params: 4,683,331
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 1.17
# Forward/backward pass size (MB): 696.41
# Params size (MB): 17.87
# Estimated Total Size (MB): 715.45
# ----------------------------------------------------------------
# Inference time on cpu: 83.00 [ms]
# Inference fps on cpu: 12.05 [fps]
# Inference time on cuda: 12.35 [ms]
# Inference fps on cuda: 80.99 [fps]

# JSEGNET21V2
# Total params: 2,697,790
# Trainable params: 2,697,790
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 1.17
# Forward/backward pass size (MB): 99.32
# Params size (MB): 10.29
# Estimated Total Size (MB): 110.78
# ----------------------------------------------------------------
# Inference time on cpu: 22.37 [ms]
# Inference fps on cpu: 44.70 [fps]
# Inference time on cuda: 5.11 [ms]
# Inference fps on cuda: 195.74 [fps]

