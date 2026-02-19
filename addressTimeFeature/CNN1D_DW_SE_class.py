from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ExtendedNNModule import ExtendedNNModule
from torch.utils.data import TensorDataset, DataLoader
from nnModule import SEBlock

# -------------------------
# 主模型：加入 SE 注意力
# -------------------------
class CNN1D_DW_SE_class(ExtendedNNModule):
	def __init__(self, addressTimeFeature_dataSet,
				 cnn_kernel_height=BTNHGV2ParameterClass.cnn_kernel_height,
				 dropout_rate=BTNHGV2ParameterClass.dropout):

		super().__init__()

		self.feature_dim = addressTimeFeature_dataSet.tensors[0].shape[-1]  # D
		self.seq_len = addressTimeFeature_dataSet.tensors[0].shape[-2]      # T=64
		self.num_classes = addressTimeFeature_dataSet.tensors[1].unique().numel()

		self.cnn_in_channels = self.feature_dim

		# Depthwise Conv
		self.conv1 = nn.Conv1d(
			in_channels=self.cnn_in_channels,
			out_channels=self.cnn_in_channels,
			kernel_size=cnn_kernel_height,
			padding="same",
			groups=self.cnn_in_channels
		)
		self.bn1 = nn.BatchNorm1d(self.cnn_in_channels)
		# self.bn2 = nn.BatchNorm1d(self.cnn_in_channels)

		# SE 通道注意力
		self.se = SEBlock(self.cnn_in_channels, reduction=4)

		self.dropout = nn.Dropout(dropout_rate)

		# Flatten 后维度
		flattened_size = self.cnn_in_channels * self.seq_len
		self.fc_out = nn.Linear(flattened_size, self.num_classes)

	def forward(self, x):
		# [B, T, D] → [B, D, T]
		x = x.permute(0, 2, 1)

		# identity = self.bn2(x)  # ★ 残差分支

		# DW + BN + ReLU
		out = self.bn1(self.conv1(x))
		out = F.relu(out)
		out = self.dropout(out)

		# SE 注意力
		out = self.se(out)

		# # ★ 残差相加
		# out = out + identity

		# 分类头
		out = torch.flatten(out, 1)
		out = self.fc_out(self.dropout(out))

		return out


# # -------------------------
# # SE 通道注意力模块
# # -------------------------
# class SEBlock(nn.Module):
# 	def __init__(self, channels, reduction=4):
# 		super().__init__()
# 		hidden = max(1, channels // reduction)

# 		self.avg_pool = nn.AdaptiveAvgPool1d(1)  # [B, C, T] → [B, C, 1]
# 		self.fc = nn.Sequential(
# 			nn.Linear(channels, hidden),
# 			nn.ReLU(inplace=True),
# 			nn.Linear(hidden, channels),
# 			nn.Sigmoid()
# 		)

# 	def forward(self, x):
# 		b, c, t = x.size()  # [B, C, T]

# 		y = self.avg_pool(x).view(b, c)  # Squeeze：[B, C, T] → [B, C]

# 		y = self.fc(y)  # Excitation：[B, C] → [B, C]（学习通道权重）

# 		y = y.view(b, c, 1)  # 重塑维度：[B, C] → [B, C, 1]

# 		return x * y  # Scale：将通道权重应用到原始特征图