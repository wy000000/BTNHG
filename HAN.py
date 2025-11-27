import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv
from torch_geometric.loader import NeighborLoader
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass

class HAN(torch.nn.Module):	
	def __init__(self, heteroDataCls: BTNHGV2HeteroDataClass,
				# accumulation_steps=BTNHGV2ParameterClass.accumulation_steps,
				hidden_channels=BTNHGV2ParameterClass.hidden_channels,
				out_channels=BTNHGV2ParameterClass.out_channels,
				num_heads=BTNHGV2ParameterClass.num_heads,
				dropout=BTNHGV2ParameterClass.dropout,
				batch_size=BTNHGV2ParameterClass.batch_size,
				shuffle=BTNHGV2ParameterClass.shuffle,
				isResetSeed=BTNHGV2ParameterClass.isResetSeed
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
			isResetSeed=BTNHGV2ParameterClass.isResetSeed
		"""
		super().__init__()
		self.heteroDataCls = heteroDataCls
		# self.train_mask, self.test_mask = \
		self.heteroDataCls.getTrainTestMask()
		self.heteroData = heteroDataCls.heteroData
		# self.accumulation_steps = accumulation_steps
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.isResetSeed = isResetSeed
		self._metadata = self.heteroData.metadata()
		#统计self.heteroData["address"].y中不等于-1的类别的数量
		self._num_classes = self.heteroData["address"].y.unique().numel()-1
		self._hidden_channels = hidden_channels
		self._out_channels = out_channels
		self._num_heads = num_heads
		self._dropout = dropout		

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
		self.ln1 = torch.nn.LayerNorm(self._hidden_channels)
		self.ln2 = torch.nn.LayerNorm(self._out_channels)

		self.lin = torch.nn.Linear(self._out_channels, self._num_classes)

	def forward(self, heteroData):
		x_dict = heteroData.collect("x")
		edge_index_dict = heteroData.collect("edge_index")
		# 第一次卷积：对每种关系做注意力聚合
		# print(f"x_dict[address]: {x_dict["address"].shape}")
		x_dict = self.conv1(x_dict, edge_index_dict)
		# print(f"x_dict after conv1: {x_dict["address"].shape}")
		x_dict = {k: self.ln1(v) for k, v in x_dict.items()}
		x_dict = {k: F.relu(v) for k, v in x_dict.items()}
		# Dropout 防止过拟合
		x_dict = {k: F.dropout(v, p=self._dropout, training=self.training) for k, v in x_dict.items()}
		# 第二次卷积
		x_dict = self.conv2(x_dict, edge_index_dict)
		# print(f"x_dict[address] after conv2: {x_dict["address"].shape}")
		x_dict = {k: self.ln2(v) for k, v in x_dict.items()}
		x_dict = {k: F.relu(v) for k, v in x_dict.items()}
		# print(f"x_dict[address]: {x_dict["address"].shape}")
		# 只对 address 节点做分类
		out=self.lin(x_dict["address"])
		#打印out的shape和类型
		# print(f"out: {out.shape}, {out.dtype}")
		return out