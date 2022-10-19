import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.ResNet import BasicConv

# CBAM 
class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
	def __init__(self, gate_channels, compressed_channels=None):
		super(ChannelGate, self).__init__()
		if compressed_channels is None:
			compressed_channels = gate_channels // 2
		self.flat = Flatten()
		self.lin1 = nn.Linear(gate_channels, compressed_channels)
		self.relu = nn.ReLU()
		self.lin2 = nn.Linear(compressed_channels, gate_channels)
	def forward(self, x, sem_encod):
		avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
		channel_att_sum = self.lin1(self.flat(avg_pool))
		if sem_encod is not None:
			channel_att_sum = channel_att_sum + sem_encod

		channel_att_sum = self.lin2(self.relu(channel_att_sum))
		scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
		return x * scale

class ChannelPool(nn.Module):
	def forward(self, x):
		GAP = torch.mean(x, dim=1, keepdim=True)
		GMP, _ = torch.max(x, dim=1, keepdim=True)
		return torch.cat([GAP, GMP], dim=1)

class SpatialGate(nn.Module):
	def __init__(self):
		super(SpatialGate, self).__init__()
		kernel_size = 7
		self.compress = ChannelPool()
		self.spatial = BasicConv(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, act=None)
	def forward(self, x):
		x_compress = self.compress(x)
		x_out = self.spatial(x_compress)
		scale = torch.sigmoid(x_out)  
		return x * scale

class CBAMResidualShared(nn.Module):

	def __init__(self, spatial_dim, semantic_dim, semantic_assert):
		super(CBAMResidualShared, self).__init__()
		self.ChannelGate = ChannelGate(semantic_dim, semantic_assert)
		self.SpatialGate = SpatialGate()
		self.ConvFusion = nn.Conv2d(spatial_dim + semantic_dim, semantic_dim, 1)
		self.ConvAttentFusion = nn.Conv2d(spatial_dim+ semantic_dim, semantic_dim, 1)

	def forward(self, spatial, semantic, sem_encod):
		spatial_ = self.SpatialGate(spatial)
		semantic_ = self.ChannelGate(semantic, sem_encod.squeeze(2)) # [b, c] + [b, c, 1] => [b, c] + [b, c]

		feature_sum = spatial_ + semantic_
		feature_prod = spatial_ * semantic_

		x_out = self.ConvFusion(torch.cat([spatial, semantic], 1))
		x_att = self.ConvAttentFusion(torch.cat([feature_sum, feature_prod], 1))

		return x_out + x_att

# Scene Aware Activation (SAA) Module
class SAA(nn.Module): 
	def __init__(self, spatial_dim, semantic_dim, semantic_assert):
		super(SAA, self).__init__()
		self.projection = nn.Conv2d(spatial_dim, semantic_dim, 1)
		self.cbam = CBAMResidualShared(semantic_dim, semantic_dim, semantic_assert)
	
	def forward(self, spatial_feature, semantic_feature, sem_encod):
		spatial_feature_proj = self.projection(spatial_feature)
		activated = self.cbam(spatial_feature_proj, semantic_feature, sem_encod)

		return activated
