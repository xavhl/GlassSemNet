import torch
import torch.nn as nn

from model.backbone.DeepLab import DeepLabHeadV3Plus
import model.backbone.ResNet_def as resnet

class BasicConv(nn.Module): ## debug
	def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, 
	pooling=None, groups=1, batchnorm=True, act=nn.ReLU(inplace=True)):
		super(BasicConv, self).__init__()
		self.layers = nn.ModuleList([
			nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias), # training acceleration: bias off before BN
		])

		if pooling:
			self.layers.append(pooling)
		
		if batchnorm:
			self.layers.append(nn.BatchNorm2d(out_planes))

		if act:
			self.layers.append(act)

		self.sequential = nn.Sequential(*self.layers)

	def forward(self, x):
		return self.sequential(x)

class ResNet_Proj(nn.Module):
	def __init__(self, channels=(1024,2048)):
		super(ResNet_Proj, self).__init__()
		layer2_channel, layer3_channel = channels
		self.conv_bn_relu1 = BasicConv(layer2_channel, layer2_channel, pooling=nn.AdaptiveAvgPool2d(24))
		self.conv_bn_relu2 = BasicConv(layer3_channel, layer3_channel, pooling=nn.AdaptiveAvgPool2d(12))

	def forward(self, semantic_feat):
		semantic_feat[2] = self.conv_bn_relu1(semantic_feat[2])
		semantic_feat[3] = self.conv_bn_relu2(semantic_feat[3])
		return semantic_feat

class Res_DeepLabV3P(nn.Module):
	'''
	Res backbone for semantic feature extraction
	'''
	def __init__(self, num_classes) -> None:
		super(Res_DeepLabV3P, self).__init__()
		
		self.resnext = resnet.__dict__['resnext50_32x4d'](pretrained=True,replace_stride_with_dilation=[False, True, True])
		self.conv1 = self.resnext.conv1
		self.bn1 = self.resnext.bn1
		self.relu = self.resnext.relu
		self.max_pool = self.resnext.maxpool
		self.layer1 = self.resnext.layer1
		self.layer2 = self.resnext.layer2
		self.layer3 = self.resnext.layer3
		self.layer4 = self.resnext.layer4

		self.projection = ResNet_Proj()
		
		self.classifier = DeepLabHeadV3Plus(num_classes=num_classes)

	def forward(self, x):
		
		features = []
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.max_pool(x); layer0 = x
		x = self.layer1(x) ; features.append(x)#; layer1 = x
		x = self.layer2(x) ; features.append(x)
		x = self.layer3(x) ; features.append(x)
		x = self.layer4(x) ; features.append(x)#; layer4 = x

		predict = self.classifier({'low_level': features[0], 'out': features[3]})
		features_ = self.projection(features)

		return {'layer0': layer0, 'backbone': features_, 'pred': predict}
