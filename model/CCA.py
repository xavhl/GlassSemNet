import torch.nn as nn
from einops import rearrange
import math


## qkv prep
class SqueezeAndExcitationSEShared(nn.Module): 
	def __init__(self, channel, activation=nn.ReLU(), channel_mid=None):
		super(SqueezeAndExcitationSEShared, self).__init__()
		if channel_mid is None:
			channel_mid = channel // 2

		self.avgpool1d = nn.AdaptiveAvgPool1d(1)
		self.conv1d1 = nn.Conv1d(channel, channel_mid, kernel_size=1)
		self.act = activation
		self.conv1d2 = nn.Conv1d(channel_mid, channel, kernel_size=1)
		self.sigm = nn.Sigmoid()

	def forward(self, rgb, sem_encod):
		weighting = self.conv1d1(self.avgpool1d(rgb))
		if sem_encod is not None:
			weighting = weighting + sem_encod
		
		weighting = self.sigm(self.conv1d2(self.act(weighting)))

		y = rgb * weighting
		return y

class Semantic_To_KV(nn.Module):
	def __init__(self, embed_dim, semantic_assert):
		super(Semantic_To_KV, self).__init__()
		self.projection = SqueezeAndExcitationSEShared(embed_dim, channel_mid=semantic_assert)
	
	def forward(self, semantic_feature, sem_encod):
		sem = rearrange(self.projection(rearrange(semantic_feature, 'hw b c -> b c hw'), sem_encod), 'b c hw -> hw b c')
		sem_k, sem_v = sem.chunk(2, dim=-1)

		return sem_k, sem_v

## Attention block 
class PreNorm(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.norm = nn.LayerNorm(dim)
		self.fn = fn
	def forward(self, x, **kwargs):
		return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
	def __init__(self, dim, hidden_dim, dropout = 0.):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, dim),
			nn.Dropout(dropout)
		)
	def forward(self, x):
		return self.net(x)

class Attention(nn.Module):
	def __init__(self, embed_dim):
		super(Attention, self).__init__()

		self.attn = nn.MultiheadAttention(embed_dim, num_heads=8)
		self.norm = nn.LayerNorm(embed_dim)
		self.ff = PreNorm(embed_dim, FeedForward(embed_dim, 1024))
		
	def forward(self, spatial_q, semantic_k, semantic_v):
		attention, _ = self.attn(spatial_q, semantic_k, semantic_v, need_weights=False)
		attention = self.norm(attention)
		attention = self.ff(attention)
		attention = attention + spatial_q
		return attention
	
# Context Correlation Attention (CCA) Module
class CCA(nn.Module): 
	def __init__(self, spatial_dim, semantic_dim, transform_dim, semantic_assert):
		super(CCA, self).__init__()

		self.spatial_to_q = nn.Linear(spatial_dim, transform_dim)
		self.semantic_to_kv = Semantic_To_KV(semantic_dim, semantic_assert)
		self.attention = Attention(transform_dim)

	def unflatten(self, x):
		return rearrange(x, '(h w) b c -> b c h w', h=int(math.sqrt(x.shape[0])))
	
	def forward(self, spatial_feature, semantic_feature, sem_encod):
		spatial = rearrange(spatial_feature, 'b c h w -> (h w) b c')
		spatial_q = self.spatial_to_q(spatial)

		semantic =  rearrange(semantic_feature, 'b c h w -> (h w) b c')
		semantic_k, semantic_v = self.semantic_to_kv(semantic, sem_encod)
		
		attended = self.attention(spatial_q, semantic_k, semantic_v)

		return self.unflatten(attended)
