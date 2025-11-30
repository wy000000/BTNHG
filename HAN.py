import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv
from torch_geometric.loader import NeighborLoader
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass

class HANClass(torch.nn.Module):	
	def __init__(self, heteroDataCls: BTNHGV2HeteroDataClass,
				hidden_channels=BTNHGV2ParameterClass.hidden_channels,
				out_channels=BTNHGV2ParameterClass.out_channels,
				num_heads=BTNHGV2ParameterClass.num_heads,
				dropout=BTNHGV2ParameterClass.dropout,
				batch_size=BTNHGV2ParameterClass.batch_size,
				shuffle=BTNHGV2ParameterClass.shuffle,
				resetSeed=BTNHGV2ParameterClass.resetSeed
				):
		"""
		Heterogeneous Graph Attention Network		
		Args:
			hidden_channels=BTNHGV2ParameterClass.hidden_channels,
			out_channels=BTNHGV2ParameterClass.out_channels,
			num_heads=BTNHGV2ParameterClass.num_heads,
			dropout=BTNHGV2ParameterClass.dropout,
			batch_size=BTNHGV2ParameterClass.batch_size,
			shuffle=BTNHGV2ParameterClass.shuffle,
			resetSeed=BTNHGV2ParameterClass.resetSeed
		"""
		super().__init__()
		self.heteroDataCls = heteroDataCls
		# self.train_mask, self.test_mask = \
		self.heteroDataCls.getTrainTestMask()
		self.heteroData = heteroDataCls.heteroData
		# self.accumulation_steps = accumulation_steps
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.resetSeed = resetSeed
		self._metadata = self.heteroData.metadata()
		#统计self.heteroData["address"].y中不等于-1的类别的数量
		self._num_classes = self.heteroData["address"].y.unique().numel()-1
		self._hidden_channels = hidden_channels
		self._out_channels = out_channels
		self._num_heads = num_heads
		self._dropout = nn.Dropout(p=dropout)

		self.conv1 = HANConv(
			in_channels=-1,
			out_channels=self._hidden_channels,
			heads=self._num_heads,
			metadata=self._metadata
		)
		self.conv2 = HANConv(
			in_channels=self._hidden_channels,
			out_channels=self._out_channels,
			heads=self._num_heads,
			metadata=self._metadata
		)
		# 归一化层：LayerNorm
		self.lynm1 = torch.nn.LayerNorm(self._hidden_channels)
		self.lynm2 = torch.nn.LayerNorm(self._out_channels)

		self.lin = torch.nn.Linear(self._out_channels, self._num_classes)

	def forward(self, heteroData):
		x_dict = heteroData.x_dict
		edge_index_dict = heteroData.edge_index_dict

		# 第一次卷积
		h1 = self.conv1(x_dict, edge_index_dict)
		# # 残差连接
		# h1 = {k: h1[k] + x_dict[k] for k in h1}
		# 归一化
		h1 = {k: self.lynm1(v) for k, v in h1.items()}
		# 激活函数
		h1 = {k: F.relu(v) for k, v in h1.items()}
		# Dropout
		h1 = {k: self._dropout(v) for k, v in h1.items()}

		# 第二次卷积
		h2 = self.conv2(h1, edge_index_dict)
		# # 残差连接
		# h2 = {k: h2[k] + h1[k] for k in h2}
		# 归一化
		h2 = {k: self.lynm2(v) for k, v in h2.items()}
		# 激活函数
		h2 = {k: F.relu(v) for k, v in h2.items()}
		# Dropout
		h2 = {k: self._dropout(v) for k, v in h2.items()}

		# 只对 address 节点做分类
		target_h = h2["address"]
		# 分类头（一般不加 ReLU/Norm/Dropout，直接输出 logits）
		out = self.lin(target_h)

		return out
