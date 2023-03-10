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
import shutil
import json
import base64
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

	parser.add_argument('--checkpoint', type=str, default=r"/media/byd/A264AC9264AC6AAD/DataSet/10_14_data/data_B/mmsegmentation_runs/iter_192000.pth",
						help='Path to the trained model file')

	parser.add_argument('--img', type=str, default=r"D:\data\cha_data\not_pipeline\jpg",
						help='Path to the input img')

	parser.add_argument('--mask', type=str, default=r"D:\data\cha_data\not_pipeline\png",
						help='Path to the output img')
	parser.add_argument('--save', type=str, default=r'D:\data\cha_data\not_pipeline\new')#paint_line_img, check_line_img
	parser.add_argument('--json', type=str, default=r'D:\data\cha_data\not_pipeline\json')#paint_line_img, check_line_img

	args = parser.parse_args()
	return args


def main():
	args = parser_arg()
	# img input
	imgs_dir = args.img
	mask_labels_dir = args.mask
	save_dir = args.save
	json_dir = args.json
	cv2.waitKey()

	imgs_dir_list = os.listdir(imgs_dir)
	for indx, img_name in enumerate(imgs_dir_list):
		img_dir = imgs_dir + '/' +img_name
		mask_dir = mask_labels_dir + '/' + img_name.replace('.jpg', '.png')

		image = cv2.imread(img_dir)
		label_mask = cv2.imread(mask_dir)
		label_mask = cv2.resize(label_mask, (512, 512))


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
		bel = mask > 2
		fg_inmg = np.zeros(shape=image.shape, dtype=np.uint8)
		fg_inmg2 = np.ones(shape=image.shape, dtype=np.uint8)*255
		FG_img = np.where(bel,fg_inmg,fg_inmg2)
		FG_img = cv2.cvtColor(FG_img, cv2.COLOR_BGR2GRAY)
		label_mask = cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY)
		ret, binary = cv2.threshold(FG_img, 125, 255, cv2.THRESH_BINARY)
		ret_label, binary_label = cv2.threshold(label_mask, 125, 255, cv2.THRESH_BINARY)

		# binary_label_f = np.array(binary_label, dtype= float)
		# binary_f = np.array(binary, dtype= float)
		# he_img = binary_label_f + binary_f
		# he_img[he_img<=300.0] = 0
		# he_img[he_img>300.0] = 255
		# he_img = np.array(he_img, dtype=np.uint8)

		# label_are = img_area(binary_label)
		# right_are = img_area(he_img)
		# if label_are == 0:
		# 	iou = 0
		# else:
		# 	iou = round((right_are/label_are), 3)

		# binary_label - binary表示label里面有，但是没有预测出来，原因是标错或者不清晰，需要人为修改标签
		# binary - binary_label表示预测有，但是标签里没有，原因是模型预测错或者漏标，需要人为将漏标的标签补上
		cha_img = binary_label - binary
		cha_img[cha_img<0] = 0
		chaimg = np.array(cha_img,dtype=np.uint8)
		chaimga = open_img(chaimg)
		img_area = compute_area(chaimga, image)
		cv2.imwrite(save_dir + '/' + img_name, img_area)
		print('img......{}......{}'.format(indx, str(img_name)))
		if os.path.exists(json_dir + '/' + img_name.replace('.jpg', '.json')):
			create_json(json_dir, img_area, img_name, save_dir)
		# if iou >= 0.5:
		# 	shutil.copy(img_dir, save_dir + '/' + img_name)

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
			# 通过调整thickness值改变红框大小
			cv2.drawContours(img_one, c, -1, 255, thickness=13)
	conts_one, hieth_one = cv2.findContours(img_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for c_one in conts_one:
		cv2.drawContours(img_two, c_one, -1, 255, -1)
	conts_two, hieth_two = cv2.findContours(img_two, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for c_two in conts_two:
		cv2.drawContours(origin_image, c_two, -1, (0, 0, 255), thickness=1)
	return origin_image

def img_area(img):
	conts, hieth = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	are_all = 0
	for c in conts:
		are = cv2.contourArea(c)
		are_all += are
	return are_all

def create_json(json_dir, img, img_name, save_dir):

	json_one_dir = json_dir + '/' + img_name.replace('.jpg', '.json')
	with open(json_one_dir, 'r', encoding='utf-8') as f:
		sett = json.load(f)
	# img_hight = sett['imageHeight']
	# img_width = sett['imageWidth']
	base64_str = sett['imageData']
	byte_data = base64.b64decode(base64_str)
	encode_image = np.asarray(bytearray(byte_data), dtype="uint8")  #
	img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)  #
	img_hight = img_array.shape[1]
	img_width = img_array.shape[0]
	# img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # BGR2RGB
	# cv2.imwrite('1.jpg', img_array)
	img_ = cv2.resize(img, (img_hight, img_width))
	cv2.imwrite(save_dir + '/' + img_name, img_)
	seet1 = copy.deepcopy(sett)
	with open(save_dir + '/' + img_name, 'rb') as g:
		img_data = g.read()
		imageData = base64.b64encode(img_data).decode('utf-8')
	seet1['imageData'] = imageData
	with open(save_dir + '/' + img_name.replace('.jpg','.json'), 'w', encoding='utf-8') as f:
		f.write(json.dumps(seet1, indent=2, ensure_ascii=False))
	print('write json .....', img_name)

if __name__=='__main__':
	main()