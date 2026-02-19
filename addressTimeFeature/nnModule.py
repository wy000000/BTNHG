import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# SE 通道注意力模块
# -------------------------
class SEBlock(nn.Module):
	def __init__(self, channels, reduction=4):
		super().__init__()
		hidden = max(1, channels // reduction)

		self.avg_pool = nn.AdaptiveAvgPool1d(1)  # [B, C, T] → [B, C, 1]
		self.fc = nn.Sequential(
			nn.Linear(channels, hidden),
			nn.ReLU(inplace=True),
			nn.Linear(hidden, channels),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, t = x.size()  # [B, C, T]

		y = self.avg_pool(x).view(b, c)  # Squeeze：[B, C, T] → [B, C]

		y = self.fc(y)  # Excitation：[B, C] → [B, C]（学习通道权重）

		y = y.view(b, c, 1)  # 重塑维度：[B, C] → [B, C, 1]

		return x * y  # Scale：将通道权重应用到原始特征图

class ConvPE(nn.Module):
	def __init__(self, channels, kernel_size=3):
		super().__init__()
		self.convPE = nn.Conv1d(
			channels, channels,
			kernel_size=kernel_size,
			padding=kernel_size // 2,
			groups=channels
		)

	def forward(self, x):
		# x: Conv1d 要求 [B, C, L]
		# return x + self.conv(x.transpose(1, 2)).transpose(1, 2)
		return x + self.convPE(x)


class MambaBlock(nn.Module):
	def __init__(self, d_model):
		super().__init__()
		self.in_proj = nn.Linear(d_model, d_model * 2)
		self.out_proj = nn.Linear(d_model, d_model)

		# SSM 核心参数
		self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
		self.act = nn.SiLU()

	def forward(self, x):
		# x: [B, T, D]
		u, v = self.in_proj(x).chunk(2, dim=-1)

		# depthwise conv for SSM
		y = self.conv1d(v.transpose(1, 2)).transpose(1, 2)

		y = self.act(y) * u
		y = self.out_proj(y)
		return y
