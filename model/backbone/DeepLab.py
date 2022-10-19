# from PyTorch's torchvision

import torch
from torch import nn
from torch.nn import functional as F

class ASPPConv(nn.Sequential):
	def __init__(self, in_channels, out_channels, dilation):
		modules = [
			nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		]
		super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
	def __init__(self, in_channels, out_channels):
		super(ASPPPooling, self).__init__(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_channels, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True))

	def forward(self, x):
		size = x.shape[-2:]
		x = super(ASPPPooling, self).forward(x)
		return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
	def __init__(self, in_channels, atrous_rates):
		super(ASPP, self).__init__()
		out_channels = 256
		modules = []
		modules.append(nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)))

		rate1, rate2, rate3 = tuple(atrous_rates)
		modules.append(ASPPConv(in_channels, out_channels, rate1))
		modules.append(ASPPConv(in_channels, out_channels, rate2))
		modules.append(ASPPConv(in_channels, out_channels, rate3))
		modules.append(ASPPPooling(in_channels, out_channels))

		self.convs = nn.ModuleList(modules)

		self.project = nn.Sequential(
			nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),)

	def forward(self, x):
		res = []
		for conv in self.convs:
			res.append(conv(x))
		res = torch.cat(res, dim=1)
		return self.project(res)

class DeepLabHeadV3Plus(nn.Module):
	def __init__(self, in_channels=2048, low_level_channels=256, num_classes=43, aspp_dilate=[12, 24, 36], output_size=None):
		super(DeepLabHeadV3Plus, self).__init__()
		self.project = nn.Sequential( 
			nn.Conv2d(low_level_channels, 48, 1, bias=False),
			nn.BatchNorm2d(48),
			nn.ReLU(inplace=True),
		)
		self.output_size = output_size

		self.aspp = ASPP(in_channels, aspp_dilate)

		self.classifier = nn.Sequential(
			nn.Conv2d(304, 256, 3, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, num_classes, 1)
		)
		self._init_weight()

	def forward(self, feature):
		low_level_feature = self.project(feature['low_level'])
		output_feature = self.aspp(feature['out'])
		output_size = self.output_size if self.output_size else low_level_feature.shape[2:]
		low_level_feature = F.interpolate(low_level_feature, size=output_size, mode='bilinear', align_corners=False)
		output_feature = F.interpolate(output_feature, size=output_size, mode='bilinear', align_corners=False)
		return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
	
	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
