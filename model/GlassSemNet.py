import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.ResNet import Res_DeepLabV3P
from model.backbone.SegFormer import SegFormer
from model.backbone.DeepLab import DeepLabHeadV3Plus
from model.UperNet import UPerNet
from model.SAA import SAA
from model.CCA import CCA

class Sem_Enc(nn.Module):
	'''
	Convert semantic features from ResNet backbone to encodings
	Input: ResNet50 [l1, l2, l3, l4]
	Output: encodings b x num_classes x 1 x 1
	'''
	def __init__(self, num_classes):
		super(Sem_Enc, self).__init__()
		
		self.projection = DeepLabHeadV3Plus(num_classes=num_classes)

		self.conv1 = nn.Conv2d(num_classes, num_classes, 7, 1, 4, groups=num_classes, bias=False) 
		self.pool1 = nn.AvgPool2d(6)

		self.conv2 = nn.Conv2d(num_classes, num_classes, 5, 1, 2, groups=num_classes, bias=False)
		self.pool2 = nn.AvgPool2d(4) # 16 => 4

		self.conv3 = nn.Conv2d(num_classes, num_classes, 4, groups=num_classes, bias=False) # 4 => 1

		self.bn = nn.BatchNorm2d(num_classes)
		self.relu = nn.ReLU()

	def forward(self, features):
		x = self.projection({'low_level': features[0], 'out': features[3]})

		conv = self.conv1(x) 
		conv = self.pool1(conv)
		conv = self.bn(conv)
		conv = self.relu(conv)

		conv = self.conv2(conv)
		conv = self.pool2(conv)
		conv = self.bn(conv)
		conv = self.relu(conv)

		conv = self.conv3(conv)
		conv = self.bn(conv)
		conv = self.relu(conv)

		return conv.squeeze(3)

class GlassSemNet(nn.Module): 

	def __init__(self):

		super(GlassSemNet, self).__init__()
		self.num_classes = 43

		self.spatial_backbone = SegFormer()
		self.semantic_backbone = Res_DeepLabV3P()
		self.sem_enc = Sem_Enc(self.num_classes)

		self.saa0 = SAA(64, 256, self.num_classes)
		self.saa1 = SAA(128, 512, self.num_classes)
		self.saa2 = SAA(320, 1024, self.num_classes)
		self.cca3 = CCA(512, 2048, 1024, self.num_classes)

		self.aux1 = nn.Conv2d(512, 1, 1)
		self.aux2 = nn.Conv2d(1024, 1, 1)
		self.decoder = UPerNet()

	def forward(self,x):

		# Spatial backbone
		spatial_feats = self.spatial_backbone(x)

		# Semantic backbone
		resnet_out = self.semantic_backbone(x)
		semantic_feats = resnet_out['backbone']
		semantic_lowlevel = resnet_out['layer0']

		# Semantic encodings
		sem_enc = self.sem_enc(semantic_feats)

		# SAA Module
		saa0 = self.saa0(spatial_feats[0], semantic_feats[0], sem_enc)
		saa1 = self.saa1(spatial_feats[1], semantic_feats[1], sem_enc)
		saa2 = self.saa2(spatial_feats[2], semantic_feats[2], sem_enc)
		
		# CCA Module
		cca3 = self.cca3(spatial_feats[3], semantic_feats[3], sem_enc)

		# Decodeer
		l0 = torch.cat([spatial_feats[0], semantic_feats[0], saa0], 1)
		l1 = torch.cat([spatial_feats[1], semantic_feats[1], saa1], 1)
		l2 = torch.cat([spatial_feats[2], semantic_feats[2], saa2], 1)
		l3 = torch.cat([spatial_feats[3], semantic_feats[3], cca3], 1)
		decoder_feats = [semantic_lowlevel, l0, l1, l2, l3]
		out = self.decoder(decoder_feats)#, self.aux1(saa1), self.aux2(cca3)

		return out
		
# if __name__ == '__main__':
# 	x = torch.rand(2,3,384,384)
# 	model = GlassSemNet()
# 	out = model(x)
# 	print(out.shape)