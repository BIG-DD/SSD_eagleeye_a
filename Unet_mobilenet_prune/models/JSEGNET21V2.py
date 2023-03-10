import torch
import torch.nn as nn
from base import BaseModel

rate = 0.5
class JSEGNET21V2(BaseModel):
	def __init__(self, num_classes=2, backbone="mobilenetv2", pretrained_backbone=None):
		super(JSEGNET21V2, self).__init__()
		self.conv1a = nn.Conv2d(3, int(32*rate), 5, stride=2, padding=2, groups=1, bias=False)
		self.bn1a = nn.BatchNorm2d(int(32*rate))
		self.act1a = nn.ReLU(inplace=True)

		self.conv1b = nn.Conv2d(int(32*rate), int(32*rate), 3, stride=1, padding=1, groups=4, bias=False)
		self.bn1b = nn.BatchNorm2d(int(32*rate))
		self.act1b = nn.ReLU(inplace=True)

		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv2a = nn.Conv2d(int(32*rate), int(64*rate), 3, stride=1, padding=1, groups=1, bias=False)
		self.bn2a = nn.BatchNorm2d(int(64*rate))
		self.act2a = nn.ReLU(inplace=True)

		self.conv2b = nn.Conv2d(int(64*rate), int(64*rate), 3, stride=1, padding=1, groups=4, bias=False)
		self.bn2b = nn.BatchNorm2d(int(64*rate))
		self.act2b = nn.ReLU(inplace=True)

		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv3a = nn.Conv2d(int(64*rate), int(128*rate), 3, stride=1, padding=1, groups=1, bias=False)
		self.bn3a = nn.BatchNorm2d(int(128*rate))
		self.act3a = nn.ReLU(inplace=True)

		self.conv3b = nn.Conv2d(int(128*rate), int(128*rate), 3, stride=1, padding=1, groups=4, bias=False)
		self.bn3b = nn.BatchNorm2d(int(128*rate))
		self.act3b = nn.ReLU(inplace=True)

		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv4a = nn.Conv2d(int(128*rate), int(256*rate), 3, stride=1, padding=1, groups=1, bias=False)
		self.bn4a = nn.BatchNorm2d(int(256*rate))
		self.act4a = nn.ReLU(inplace=True)

		self.conv4b = nn.Conv2d(int(256*rate), int(256*rate), 3, stride=1, padding=1, groups=4, bias=False)
		self.bn4b = nn.BatchNorm2d(int(256*rate))
		self.act4b = nn.ReLU(inplace=True)

		self.pool4 = nn.MaxPool2d(kernel_size=1, stride=1)

		self.conv5a = nn.Conv2d(int(256*rate), int(512*rate), 3, stride=1, padding=2, dilation=2, groups=1, bias=False)
		self.bn5a = nn.BatchNorm2d(int(512*rate))
		self.act5a = nn.ReLU(inplace=True)

		self.conv5b = nn.Conv2d(int(512*rate), int(512*rate), 3, stride=1, padding=2, dilation=2, groups=4, bias=False)
		self.bn5b = nn.BatchNorm2d(int(512*rate))
		self.act5b = nn.ReLU(inplace=True)

		self.conv_out5a = nn.Conv2d(int(512*rate), int(64*rate), 3, stride=1, padding=4, dilation=4, groups=2, bias=False)
		self.bn_out5a = nn.BatchNorm2d(int(64*rate))
		self.act_out5a = nn.ReLU(inplace=True)

		self.out5a_up2 = nn.ConvTranspose2d(int(64*rate), int(64*rate), 4, stride=2, padding=1, groups=int(64*rate))

		self.conv_out3a = nn.Conv2d(int(128*rate), int(64*rate), 3, stride=1, padding=1, dilation=1, groups=2, bias=False)
		self.bn_out3a = nn.BatchNorm2d(int(64*rate))
		self.act_out3a = nn.ReLU(inplace=True)

		self.ctx_conv1 = nn.Conv2d(int(64*rate), int(64*rate), 3, stride=1, padding=1, dilation=1, groups=1, bias=False)
		self.ctx_conv1_bn = nn.BatchNorm2d(int(64*rate))
		self.ctx_conv1_act = nn.ReLU(inplace=True)

		self.ctx_conv2 = nn.Conv2d(int(64*rate), int(64*rate), 3, stride=1, padding=4, dilation=4, groups=1, bias=False)
		self.ctx_conv2_bn = nn.BatchNorm2d(int(64*rate))
		self.ctx_conv2_act = nn.ReLU(inplace=True)

		self.ctx_conv3 = nn.Conv2d(int(64*rate), int(64*rate), 3, stride=1, padding=4, dilation=4, groups=1, bias=False)
		self.ctx_conv3_bn = nn.BatchNorm2d(int(64*rate))
		self.ctx_conv3_act = nn.ReLU(inplace=True)

		self.ctx_conv4 = nn.Conv2d(int(64*rate), int(64*rate), 3, stride=1, padding=4, dilation=4, groups=1, bias=False)
		self.ctx_conv4_bn = nn.BatchNorm2d(int(64*rate))
		self.ctx_conv4_act = nn.ReLU(inplace=True)

		self.ctx_conv_final = nn.Conv2d(int(64*rate), num_classes, 3, stride=1, padding=1, dilation=1, groups=1, bias=False)
		self.ctx_conv_bn_final = nn.BatchNorm2d(num_classes)
		self.ctx_conv_act_final = nn.ReLU(inplace=True)

		self.out_deconv_final_up2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, padding=1, stride=2, groups=num_classes)
		self.out_deconv_final_up4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, padding=1, stride=2, groups=num_classes)
		self.out_deconv_final_up8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, padding=1, stride=2, groups=num_classes)

		self._init_weights()

	def forward(self, x):
		out = self.act1a(self.bn1a(self.conv1a(x)))
		out = self.act1b(self.bn1b(self.conv1b(out)))
		out = self.pool1(out)	 # 320->160

		out = self.act2a(self.bn2a(self.conv2a(out)))
		out = self.act2b(self.bn2b(self.conv2b(out)))
		out = self.pool2(out)	 # 160->80

		out = self.act3a(self.bn3a(self.conv3a(out)))
		branch = self.act3b(self.bn3b(self.conv3b(out)))
		branch1 = self.pool3(branch)	 # 80->40

		branch1 = self.act4a(self.bn4a(self.conv4a(branch1)))
		branch1 = self.act4b(self.bn4b(self.conv4b(branch1)))
		branch1 = self.pool4(branch1)	 # 40->20

		branch1 = self.act5a(self.bn5a(self.conv5a(branch1)))
		branch1 = self.act5b(self.bn5b(self.conv5b(branch1)))

		branch1 = self.act_out5a(self.bn_out5a(self.conv_out5a(branch1)))

		branch1 = self.out5a_up2(branch1)

		branch2 = self.act_out3a(self.bn_out3a(self.conv_out3a(branch)))
		out = branch1 + branch2

		out = self.ctx_conv1_act(self.ctx_conv1_bn(self.ctx_conv1(out)))

		out = self.ctx_conv2_act(self.ctx_conv2_bn(self.ctx_conv2(out)))

		out = self.ctx_conv3_act(self.ctx_conv3_bn(self.ctx_conv3(out)))

		out = self.ctx_conv4_act(self.ctx_conv4_bn(self.ctx_conv4(out)))

		out = self.ctx_conv_act_final(self.ctx_conv_bn_final(self.ctx_conv_final(out)))

		out = self.out_deconv_final_up2(out)
		out = self.out_deconv_final_up4(out)
		out = self.out_deconv_final_up8(out)

		return out

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
	model = JSEGNET21V2(2)
	print(model)
	from torchsummary import summary
	model = model.to("cuda")
	summary(model, (3, 32, 32))

# (3, 320, 320)
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