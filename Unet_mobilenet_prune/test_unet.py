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
import copy
from models import MCnet_resnet18, UNet
#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
def parser_arg():
	parser = argparse.ArgumentParser(description="Arguments for the script")

	parser.add_argument('--use_cuda', action='store_true', default=False,
						help='Use GPU acceleration')

	parser.add_argument('--bg', type=str, default=None,
						help='Path to the background image file')

	parser.add_argument('--watch', action='store_true', default=False,
						help='Indicate show result live')

	parser.add_argument('--input_sz', type=int, default=320,
						help='Input size')

	parser.add_argument('--checkpoint', type=str, default="/media/byd/A264AC9264AC6AAD/DataSet/8_25_data/unet_mobilenet_prune/parking_slot/1129_183533/141.pth",
						help='Path to the trained model file')

	parser.add_argument('--img', type=str, default="/media/byd/A264AC9264AC6AAD/DataSet/10_14_data/data_A/data_set/images/val",
						help='Path to the input img')

	parser.add_argument('--mask', type=str, default="/media/byd/A264AC9264AC6AAD/DataSet/10_14_data/data_A/mask",
						help='Path to the output img')
	parser.add_argument('--save', type=str, default='/media/byd/A264AC9264AC6AAD/DataSet/10_14_data/data_A/Unet_runs/pre_img')#paint_line_img, check_line_img

	args = parser.parse_args()
	return args


def main():
	args = parser_arg()
	# img input
	model = MCnet_resnet18()
	if args.use_cuda:
		model = model.cuda()
	trained_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
	model.load_state_dict(trained_dict, strict=False)
	model.eval()

	imgs_dir = args.img
	mask_labels_dir = args.mask
	save_dir = args.save

	imgs_dir_list = os.listdir(imgs_dir)
	for indx, img_name in enumerate(imgs_dir_list):

		img_dir = imgs_dir + '/' +img_name
		mask_dir = mask_labels_dir + '/' + img_name.replace('.jpg', '.png')

		image = cv2.imread(img_dir)
		label_mask = cv2.imread(mask_dir)


		#------------------------------------------------------------------------------
		#	Create model and load weights
		#------------------------------------------------------------------------------
		# model = UNet(
		# 	backbone="mobilenetv2",
		# 	num_classes=2,
		# 	pretrained_backbone=None
		# )


		h, w = image.shape[0], image.shape[1]

		# Predict mask
		X, pad_up, pad_left, h_new, w_new = utils.preprocessing(image, expected_size=args.input_sz, pad_value=0)
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

		image_alpha, mask = utils.draw_matting(image, mask)
		only_save_image = True
		if only_save_image:
			cv2.imwrite(save_dir + '/' + img_name, image_alpha)

		else:
			bel = mask > 2
			fg_inmg = np.zeros(shape=image.shape, dtype=np.uint8)
			fg_inmg2 = np.ones(shape=image.shape, dtype=np.uint8)*255
			FG_img = np.where(bel,fg_inmg,fg_inmg2)
			FG_img = cv2.cvtColor(FG_img, cv2.COLOR_BGR2GRAY)
			label_mask = cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY)
			ret, binary = cv2.threshold(FG_img, 125, 255, cv2.THRESH_BINARY)
			ret_label, binary_label = cv2.threshold(label_mask, 125, 255, cv2.THRESH_BINARY)
			# binary_label - binary表示label里面有，但是没有预测出来，原因是标错或者不清晰，需要人为修改标签
			# binary - binary_label表示预测有，但是标签里没有，原因是模型预测错或者漏标，需要人为将漏标的标签补上
			cha_img = binary_label - binary
			cha_img[cha_img<0] = 0
			chaimg = np.array(cha_img,dtype=np.uint8)
			chaimga = open_img(chaimg)
			img_area = compute_area(chaimga, image)
			print('img......{}......{}'.format(indx, str(img_name)))
			cv2.imwrite(save_dir + '/' + img_name, img_area)

def open_img(img):
	k = np.ones((5, 5), np.uint8)
	close = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
	return close

def compute_area(img, origin_img):
	origin_image = copy.deepcopy(origin_img)
	ret, binary = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
	conts, hieth = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	img_one = np.zeros(shape=origin_img.shape[:-1], dtype=np.uint8)
	img_two = np.zeros(shape=origin_img.shape[:-1], dtype=np.uint8)
	for c in conts:
		are = cv2.contourArea(c)
		if are > 100:
			cv2.drawContours(img_one, c, -1, 255, thickness=13)
	conts_one, hieth_one = cv2.findContours(img_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for c_one in conts_one:
		cv2.drawContours(img_two, c_one, -1, 255, -1)
	conts_two, hieth_two = cv2.findContours(img_two, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for c_two in conts_two:
		cv2.drawContours(origin_image, c_two, -1, (0, 0, 255), thickness=1)
	return origin_image

if __name__=='__main__':
	main()