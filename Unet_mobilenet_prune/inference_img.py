#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os

import cv2, torch, argparse
from time import time
import numpy as np
from torch.nn import functional as F

from models import UNet
from dataloaders import transforms
from utils import utils


#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
def argss():
	parser = argparse.ArgumentParser(description="Arguments for the script")

	parser.add_argument('--use_cuda', action='store_true', default=False,
						help='Use GPU acceleration')

	parser.add_argument('--bg', type=str, default=None,
						help='Path to the background image file')

	parser.add_argument('--watch', action='store_true', default=False,
						help='Indicate show result live')

	parser.add_argument('--input_sz', type=int, default=320,
						help='Input size')

	parser.add_argument('--checkpoint', type=str, default="/media/byd/A264AC9264AC6AAD/DataSet/10_25_data/Unet_runs/data_enhancement/parking_slot/1027_190304/216.pth",
						help='Path to the trained model file')

	parser.add_argument('--img', type=str, default="/media/byd/A264AC9264AC6AAD/DataSet/10_25_data/Unet_runs/test_img/datas/",
						help='Path to the input img')

	parser.add_argument('--save_dir', type=str, default="/media/byd/A264AC9264AC6AAD/DataSet/10_25_data/Unet_runs/test_img/out/",
						help='segment reluts path')

	args = parser.parse_args()
	return args
def main():
	args = argss()
	img_dir = args.img
	img_list = os.listdir(img_dir)
	out_save = args.save_dir
	for img_name in img_list:
		img_path = os.path.join(img_dir, img_name)
		image = cv2.imread(img_path)
		H, W = image.shape[0], image.shape[1]

		# Video output
		# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
		# out = cv2.VideoWriter(args.output, fourcc, 30, (W,H))
		# font = cv2.FONT_HERSHEY_SIMPLEX

		# Background
		if args.bg is not None:
			BACKGROUND = cv2.imread(args.bg)[...,::-1]
			BACKGROUND = cv2.resize(BACKGROUND, (W,H), interpolation=cv2.INTER_LINEAR)
			KERNEL_SZ = 25
			SIGMA = 0

		# Alpha transperency
		else:
			COLOR1 = [255, 0, 0]
			COLOR2 = [0, 0, 255]


		#------------------------------------------------------------------------------
		#	Create model and load weights
		#------------------------------------------------------------------------------
		model = UNet(
			backbone="mobilenetv2",
			num_classes=2,
			pretrained_backbone=None
		)
		if args.use_cuda:
			model = model.cuda()
		trained_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
		model.load_state_dict(trained_dict, strict=False)
		model.eval()


		#------------------------------------------------------------------------------
		#   Predict frames
		#------------------------------------------------------------------------------
		i = 0
		#while(image is not None):
			# Read frame from camera
		start_time = time()
		h, w = image.shape[0], image.shape[1]
		read_cam_time = time()

		# Predict mask
		X, pad_up, pad_left, h_new, w_new = utils.preprocessing(image, expected_size=args.input_sz, pad_value=0)
		preproc_time = time()
		with torch.no_grad():
			if args.use_cuda:
				mask = model(X.cuda())
				mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
				mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
				mask = F.softmax(mask, dim=1)
				mask = mask[0,1,...].cpu().numpy()
			else:
				mask = model(X)
				mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
				mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
				mask = F.softmax(mask, dim=1)
				mask = mask[0,1,...].numpy()
		predict_time = time()


		# Draw result
		if args.bg is None:
			image_alpha, mask = utils.draw_matting(image, mask)
			# image_alpha = utils.draw_transperency(image, mask, COLOR1, COLOR2)
		else:
			image_alpha = utils.draw_fore_to_back(image, mask, BACKGROUND, kernel_sz=KERNEL_SZ, sigma=SIGMA)
		draw_time = time()

		# Print runtime
		read = read_cam_time-start_time
		preproc = preproc_time-read_cam_time
		pred = predict_time-preproc_time
		draw = draw_time-predict_time
		total = read + preproc + pred + draw
		fps = 1 / total
		print("read: %.3f [s]; preproc: %.3f [s]; pred: %.3f [s]; draw: %.3f [s]; total: %.3f [s]; fps: %.2f [Hz]" %
			(read, preproc, pred, draw, total, fps))
		cv2.imwrite(out_save + '/' + img_name, image_alpha)
if __name__=='__main__':
	main()