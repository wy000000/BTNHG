# import set_parent_dir
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ExtendedNNModule import ExtendedNNModule
from torch.utils.data import TensorDataset, DataLoader

class simple2DCNNClass(ExtendedNNModule):
	"""
	用于地址时间特征分类的2D卷积神经网络
	
	输入形状: [batch_size, 1, num_blocks, num_features]
	其中:
	- num_blocks: 区块数量 (5621)
	- num_features: 每个区块的特征数量 (13)
	"""
	def __init__(self,
			  		addressTimeFeature_dataSet: TensorDataset,
					# addressTimeDataCls:addressTimeDataClass,
					cnn_in_channels=1,  # 输入通道数
					cnn_out_channels=BTNHGV2ParameterClass.cnn_out_channels,  # 输出通道数
					cnn_hidden_channels=BTNHGV2ParameterClass.cnn_hidden_channels,  # 隐藏层通道数
					dropout_rate=BTNHGV2ParameterClass.dropout,
					pool_width=BTNHGV2ParameterClass.pool_width,
					pool_height=BTNHGV2ParameterClass.pool_height,
					cnn_kernel_size=BTNHGV2ParameterClass.cnn_kernel_size):  # Dropout率
		
		super().__init__()
		# self.addressTimeDataCls = addressTimeDataCls
		self.addressTimeFeature_dataSet = addressTimeFeature_dataSet
		# self.dataSet = self.addressTimeDataCls.dataSet
		# self.train_dataLoader = self.addressTimeDataCls.train_dataLoader
		# self.test_dataLoader = self.addressTimeDataCls.test_dataLoader
		# self.kfoldLoader = self.addressTimeDataCls.kFold_dataloaders
		self.feature_dim = self.addressTimeFeature_dataSet.tensors[0].shape[-1]
		self.seq_len = self.addressTimeFeature_dataSet.tensors[0].shape[-2]
		self.num_classes = self.addressTimeFeature_dataSet.tensors[1].unique().numel()

		self.cnn_in_channels = cnn_in_channels
		self.cnn_hidden_channels = cnn_hidden_channels
		self.cnn_out_channels = cnn_out_channels
		self.pool_width = pool_width
		self.pool_height = pool_height
		self.cnn_kernel_size = cnn_kernel_size		
		self.dropout_rate = dropout_rate
		
		# 卷积层1: 提取局部特征
		self.conv1 = nn.Conv2d(
			in_channels=self.cnn_in_channels, 
			out_channels=self.cnn_hidden_channels,
			kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),  # 3x3卷积核
			#padding=(1, 1)  # 保持空间维度不变
			padding='same',
			stride=1
		)
		self.bn1 = nn.BatchNorm2d(self.cnn_hidden_channels)
		
		# 池化层1: 降低空间维度
		self.pool1 = nn.AvgPool2d(kernel_size=(self.pool_height, self.pool_width))  # 沿着区块维度下采样
		
		# 卷积层2: 提取更高级别的特征
		self.conv2 = nn.Conv2d(
			in_channels=self.cnn_hidden_channels, 
			out_channels=self.cnn_out_channels, 
			kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),  # 3x3卷积核
			# padding=(1, 1)  # 保持空间维度不变
			padding='same',
			stride=1
		)
		self.bn2 = nn.BatchNorm2d(self.cnn_out_channels)
		
		# 池化层2
		self.pool2 = nn.AvgPool2d(kernel_size=(self.pool_height, self.pool_width))
		
		# Dropout层
		self.dropout = nn.Dropout(self.dropout_rate)
		
		# 计算全连接层的输入维度
		# 计算经过两次池化后的尺寸
		pooled_seq_len = self.seq_len // self.pool_height // self.pool_height  # 序列长度经过池化后的尺寸
		pooled_feature_dim = self.feature_dim // self.pool_width // self.pool_width  # 特征维度经过池化后的尺寸

		# 经过卷积和池化后的总展平尺寸
		flattened_size = self.cnn_out_channels * pooled_seq_len * pooled_feature_dim

		self.fc1 = nn.Linear(flattened_size, self.cnn_out_channels)
		
		# 输出层
		self.fc_out = nn.Linear(self.cnn_out_channels, self.num_classes)
		
	def forward(self, x):
		"""
		前向传播
		
		参数:
			x: 输入张量，形状为 [batch_size, 1, num_blocks, num_features]
			
		返回:
			output: 输出张量，形状为 [batch_size, num_classes]
		"""
	
		x = x.unsqueeze(1)
		# print(f"x shape: {x.shape}")

		# 卷积层1 -> 批量归一化 -> ReLU -> 池化
		x = self.pool1(F.relu(self.bn1(self.conv1(x))))
		
		# 卷积层2 -> 批量归一化 -> ReLU -> 池化
		x = self.pool2(F.relu(self.bn2(self.conv2(x))))		
		
		# Flatten
		x = torch.flatten(x, 1)
		
		# 全连接层1 -> Dropout
		x = self.dropout(F.relu(self.fc1(x)))
		
		# 输出层
		x = self.fc_out(x)
		
		return x

