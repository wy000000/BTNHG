import torch
import torch.nn as nn
import torch.nn.functional as F
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from ExtendedNNModule import ExtendedNNModule

class simple2DCNNClass(ExtendedNNModule):
	"""
	用于地址时间特征分类的2D卷积神经网络
	
	输入形状: [batch_size, 1, num_blocks, num_features]
	其中:
	- num_blocks: 区块数量 (5621)
	- num_features: 每个区块的特征数量 (13)
	"""
	def __init__(self, 
				 in_channels=1,  # 输入通道数
				 out_channels=BTNHGV2ParameterClass.out_channels,  # 输出通道数
				 hidden_channels=BTNHGV2ParameterClass.hidden_channels,  # 隐藏层通道数
				 num_classes=None,  # 分类数量
				 dropout_rate=BTNHGV2ParameterClass.dropout):  # Dropout率
		
		super().__init__()
		
		# 卷积层1: 提取局部特征
		self.conv1 = nn.Conv2d(
			in_channels=in_channels, 
			out_channels=hidden_channels, 
			kernel_size=(3, 3),  # 3x3卷积核
			#padding=(1, 1)  # 保持空间维度不变
			padding='same',
			stride=1
		)
		self.bn1 = nn.BatchNorm2d(hidden_channels)
		
		# 池化层1: 降低空间维度
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))  # 沿着区块维度下采样
		
		# 卷积层2: 提取更高级别的特征
		self.conv2 = nn.Conv2d(
			in_channels=hidden_channels, 
			out_channels=hidden_channels * 2, 
			kernel_size=(3, 3),
			# padding=(1, 1)  # 保持空间维度不变
			padding='same',
			stride=1
		)
		self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
		
		# 池化层2
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
		
		# Dropout层
		self.dropout = nn.Dropout(dropout_rate)
		
		# 计算全连接层的输入维度
		# 假设输入形状为 [batch_size, 1, 5621, 13]
		# 经过3次池化，区块维度变为 5621 -> 2810 -> 1405
		# 特征维度保持为13（因为池化核的第二个维度为1）
		self.fc1 = nn.Linear(hidden_channels * 2 * 1405 * 13, hidden_channels*2)
		
		# 输出层，默认使用2个分类（可以根据实际数据调整）
		self.fc_out = nn.Linear(hidden_channels*2, num_classes)
		
	def forward(self, x):
		"""
		前向传播
		
		参数:
			x: 输入张量，形状为 [batch_size, 1, num_blocks, num_features]
			
		返回:
			output: 输出张量，形状为 [batch_size, num_classes]
		"""
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

