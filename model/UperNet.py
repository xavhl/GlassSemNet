import torch
import torch.nn as nn
from model.backbone.ResNet import BasicConv

class UPerNet(nn.Module):
	def __init__(self, num_class=1,
				 use_softmax=False, pool_scales=(1, 2, 3, 6),
				 fpn_inplanes=(64, 576, 1152, 2368, 3584), fpn_dim=512):
		super(UPerNet, self).__init__()
		self.use_softmax = use_softmax

		# PPM Module
		self.ppm_pooling = []
		self.ppm_conv = []

		for scale in pool_scales:
			self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
			self.ppm_conv.append(BasicConv(fpn_inplanes[-1], 512, kernel_size=1, padding=0))
		self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
		self.ppm_conv = nn.ModuleList(self.ppm_conv)
		self.ppm_last_conv = BasicConv(fpn_inplanes[-1] + len(pool_scales)*512, fpn_dim, padding=0)

		# FPN Module
		self.fpn_in = []
		for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
			self.fpn_in.append(BasicConv(fpn_inplane, fpn_dim, kernel_size=1))
		self.fpn_in = nn.ModuleList(self.fpn_in)

		self.fpn_out = []
		for i in range(len(fpn_inplanes) - 1):  # skip the top layer
			self.fpn_out.append(nn.Sequential(
				BasicConv(fpn_dim, fpn_dim),
			))
		self.fpn_out = nn.ModuleList(self.fpn_out)

		self.conv_last = nn.Sequential(
			BasicConv(len(fpn_inplanes) * fpn_dim, fpn_dim),
			nn.Conv2d(fpn_dim, num_class, kernel_size=1)
		)

	def forward(self, conv_out):
		conv5 = conv_out[-1]

		input_size = conv5.size()
		ppm_out = [conv5]
		for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
			ppm_out.append(pool_conv(nn.functional.interpolate(
				pool_scale(conv5),
				(input_size[2], input_size[3]),
				mode='bilinear', align_corners=False)))
		ppm_out = torch.cat(ppm_out, 1)
		f = self.ppm_last_conv(ppm_out)

		fpn_feature_list = [f]
		for i in reversed(range(len(conv_out) - 1)):
			conv_x = conv_out[i]
			conv_x = self.fpn_in[i](conv_x) # lateral branch

			f = nn.functional.interpolate(
				f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
			f = conv_x + f

			fpn_feature_list.append(self.fpn_out[i](f))

		fpn_feature_list.reverse() # [P2 - P5]
		output_size = fpn_feature_list[0].size()[2:]
		fusion_list = [fpn_feature_list[0]]
		for i in range(1, len(fpn_feature_list)):
			fusion_list.append(nn.functional.interpolate(
				fpn_feature_list[i],
				output_size,
				mode='bilinear', align_corners=False))
		fusion_out = torch.cat(fusion_list, 1)
		x = self.conv_last(fusion_out)

		return x
