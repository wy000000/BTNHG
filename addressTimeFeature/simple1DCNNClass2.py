# import set_parent_dir
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ExtendedNNModule import ExtendedNNModule
from torch.utils.data import TensorDataset, DataLoader

#AdaptiveAvgPool1d(output_size)

class simple1DCNNClass(ExtendedNNModule):
	"""
	用于地址时间特征分类的1D卷积神经网络

	输入形状: [batch_size, seq_len, num_features]
	转换后形状: [batch_size, num_features, seq_len]
	"""

	def __init__(self,
				addressTimeFeature_dataSet: TensorDataset,
				#cnn_in_channels=None,  # 会自动设置为 feature_dim
				cnn_hidden_channels=BTNHGV2ParameterClass.cnn_hidden_channels,
				cnn_out_channels=BTNHGV2ParameterClass.cnn_out_channels,
				dropout_rate=BTNHGV2ParameterClass.dropout,
				pool_height=BTNHGV2ParameterClass.pool_height,  # 1D池化只需要一个维度
				cnn_kernel_height=BTNHGV2ParameterClass.cnn_kernel_height):

		super().__init__()

		self.addressTimeFeature_dataSet = addressTimeFeature_dataSet
		self.feature_dim = addressTimeFeature_dataSet.tensors[0].shape[-1]   # D = 13
		self.seq_len = addressTimeFeature_dataSet.tensors[0].shape[-2]       # T = 5621
		self.num_classes = addressTimeFeature_dataSet.tensors[1].unique().numel()

		# 1D CNN 输入通道 = 特征维度
		self.cnn_in_channels = self.feature_dim
		self.cnn_hidden_channels = cnn_hidden_channels
		self.cnn_out_channels = cnn_out_channels
		self.pool_height = pool_height
		self.cnn_kernel_height = cnn_kernel_height
		self.dropout_rate = dropout_rate

		# 卷积层1
		self.conv1 = nn.Conv1d(
			in_channels=self.cnn_in_channels,
			out_channels=self.cnn_hidden_channels,
			kernel_size=self.cnn_kernel_height,
			padding="same"
		)
		self.bn1 = nn.BatchNorm1d(self.cnn_hidden_channels)
		self.pool1 = nn.AdaptiveAvgPool1d(self.cnn_hidden_channels)

		# 卷积层2
		self.conv2 = nn.Conv1d(
			in_channels=self.cnn_hidden_channels,
			out_channels=self.cnn_out_channels,
			kernel_size=self.cnn_kernel_height,
			padding="same"
		)
		self.bn2 = nn.BatchNorm1d(self.cnn_out_channels)
		self.pool2 = nn.AvgPool1d(kernel_size=self.pool_height)

		# Dropout
		self.dropout = nn.Dropout(self.dropout_rate)

		# 计算池化后的序列长度
		pooled_seq_len = self.seq_len // self.pool_height // self.pool_height

		# Flatten 后的维度
		flattened_size = self.cnn_out_channels * pooled_seq_len

		# 全连接层
		self.fc1 = nn.Linear(flattened_size, self.cnn_out_channels)
		self.fc_out = nn.Linear(self.cnn_out_channels, self.num_classes)

	def forward(self, x):
		"""
		输入 x: [batch, seq_len, feature_dim]
		转换为: [batch, feature_dim, seq_len]
		"""

		# 转换为 1D CNN 输入格式
		x = x.permute(0, 2, 1)  # [B, D, T]

		# Conv1 -> BN -> ReLU -> Pool
		x = self.pool1(F.relu(self.bn1(self.conv1(x))))

		# Conv2 -> BN -> ReLU -> Pool
		x = self.pool2(F.relu(self.bn2(self.conv2(x))))

		# Flatten
		x = torch.flatten(x, 1)

		# FC1 -> Dropout
		x = self.dropout(F.relu(self.fc1(x)))

		# 输出层
		x = self.fc_out(x)

		return x
