from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ExtendedNNModule import ExtendedNNModule
from torch.utils.data import TensorDataset, DataLoader

class simple1DCNNClass(ExtendedNNModule):
	"""
	各通道独立卷积（Depthwise 1D CNN）
	输入: [batch, seq_len, feature_dim]
	转换: [batch, feature_dim, seq_len]
	"""

	def __init__(self,
				 addressTimeFeature_dataSet: TensorDataset,
				 cnn_hidden_channels=BTNHGV2ParameterClass.cnn_hidden_channels,
				 cnn_out_channels=BTNHGV2ParameterClass.cnn_out_channels,
				 dropout_rate=BTNHGV2ParameterClass.dropout,
				 pool_height=BTNHGV2ParameterClass.pool_height,
				 cnn_kernel_height=BTNHGV2ParameterClass.cnn_kernel_height,
				#  cnn_hidden_fc_out=BTNHGV2ParameterClass.cnn_hidden_fc_out
				 ):

		super().__init__()

		self.addressTimeFeature_dataSet = addressTimeFeature_dataSet
		self.feature_dim = addressTimeFeature_dataSet.tensors[0].shape[-1]   # D = 13
		self.seq_len = addressTimeFeature_dataSet.tensors[0].shape[-2]       # T = 5621
		self.num_classes = addressTimeFeature_dataSet.tensors[1].unique().numel()

		self.cnn_in_channels = self.feature_dim
		self.cnn_kernel_height = cnn_kernel_height
		self.dropout_rate = dropout_rate

		# -------------------------
		# Depthwise Conv1（不跨通道）
		# -------------------------
		self.conv1 = nn.Conv1d(
			in_channels=self.cnn_in_channels,
			out_channels=self.cnn_in_channels,  # 每个通道独立卷积
			kernel_size=self.cnn_kernel_height,
			padding="same",
			groups=self.cnn_in_channels        # 关键参数：各通道独立卷积
		)
		self.bn1 = nn.BatchNorm1d(self.cnn_in_channels)

		# # -------------------------
		# # Depthwise Conv2（不跨通道）
		# # -------------------------
		# self.conv2 = nn.Conv1d(
		# 	in_channels=self.cnn_in_channels,
		# 	out_channels=self.cnn_in_channels,
		# 	kernel_size=self.cnn_kernel_height,
		# 	padding="same",
		# 	groups=self.cnn_in_channels
		# )
		# self.bn2 = nn.BatchNorm1d(self.cnn_in_channels)

		# Dropout
		self.dropout = nn.Dropout(self.dropout_rate)

		# Flatten 后维度
		flattened_size = self.cnn_in_channels * self.seq_len

		# 输出层
		self.fc_out = nn.Linear(flattened_size, self.num_classes)

	def forward(self, x):
		# 输入: [B, T, D] → 转为 [B, D, T]
		x = x.permute(0, 2, 1)

		x = F.relu(self.bn1(self.conv1(x)))
		# x = F.relu(self.bn2(self.conv2(x)))

		x = torch.flatten(x, 1)
		x = self.fc_out(self.dropout(x))

		return x
