import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from ExtendedNNModule import ExtendedNNModule
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
from BTNHGV2ParameterClass import BTNHGV2ParameterClass

class SAGEClass(ExtendedNNModule):
	def __init__(self,
				heteroData: HeteroData,
				hidden_channels=BTNHGV2ParameterClass.hidden_channels,
				out_channels=BTNHGV2ParameterClass.out_channels,
				num_layers=BTNHGV2ParameterClass.num_layers,
				num_heads=BTNHGV2ParameterClass.num_heads,
				dropout=BTNHGV2ParameterClass.dropout):
		super().__init__()
		self._dropout = nn.Dropout(p=dropout)
		# self.heteroDataCls = heteroDataCls
		self.heteroData = heteroData
		self.num_heads = num_heads
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.num_layers = num_layers

		# 获取节点特征维度
		in_channels_dict = {}
		for node_type in self.heteroData.node_types:
			in_channels_dict[node_type] = self.heteroData[node_type].x.size(-1)

		# 构建多层 HeteroConv
		self.convs = nn.ModuleList()
		for i in range(self.num_layers):
			conv_dict = {}
			for edge_type in self.heteroData.edge_types:
				src, rel, dst = edge_type
				in_src = in_channels_dict[src] if i == 0 else self.hidden_channels
				in_dst = in_channels_dict[dst] if i == 0 else self.hidden_channels
				out_channels_layer = self.hidden_channels if i < self.num_layers - 1 else self.out_channels

				conv_dict[edge_type] = SAGEConv((in_src, in_dst), out_channels_layer)

			self.convs.append(HeteroConv(conv_dict, aggr='sum'))

		self.norms = nn.ModuleDict()
		for node_type in self.heteroData.node_types:
			self.norms[node_type] = nn.BatchNorm1d(hidden_channels)

	def forward(self, hetero_data):
		x_dict, edge_index_dict = hetero_data.x_dict, hetero_data.edge_index_dict

		for i, conv in enumerate(self.convs):
			x_dict = conv(x_dict, edge_index_dict)
			if i < self.num_layers - 1:
				x_dict = {key: F.relu(x) for key, x in x_dict.items()}
				# BatchNorm
				x_dict = {key: self.norms[key](x) for key, x in x_dict.items()}
				# Dropout
				x_dict = {key: self._dropout(x) for key, x in x_dict.items()}				
				# 只返回 address 节点的分类结果
		return x_dict["address"]
