import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv
from torch_geometric.loader import NeighborLoader

class HAN(torch.nn.Module):	
	def __init__(self, metadata, hidden_channels, out_channels, num_heads, dropout, num_classes):
		#给各参数加上注释
		# metadata: 节点类型和边类型的元数据
		# hidden_channels: 隐藏层通道数
		# out_channels: 输出层通道数
		# num_heads: 注意力头数
		# dropout: Dropout 概率
		# num_classes: 分类数
		super().__init__()
		self.conv1 = HANConv(
			in_channels=-1,
			out_channels=hidden_channels,
			heads=num_heads,
			metadata=metadata
		)
		self.conv2 = HANConv(
			in_channels=hidden_channels * num_heads,
			out_channels=out_channels,
			heads=num_heads,
			metadata=metadata
		)
		self.lin = torch.nn.Linear(out_channels * num_heads, num_classes)
		self.dropout = dropout

	def forward(self, x_dict, edge_index_dict):
		# 第一次卷积：对每种关系做注意力聚合
		x_dict = self.conv1(x_dict, edge_index_dict)
		x_dict = {k: F.relu(v) for k, v in x_dict.items()}
		# Dropout 防止过拟合
		x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
		# 第二次卷积
		x_dict = self.conv2(x_dict, edge_index_dict)
		x_dict = {k: F.relu(v) for k, v in x_dict.items()}
		# 只对 address 节点做分类
		return self.lin(x_dict["address"])