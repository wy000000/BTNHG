import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, GraphConv
from torch_geometric.data import HeteroData
from ExtendedNNModule import ExtendedNNModule
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
from BTNHGV2ParameterClass import BTNHGV2ParameterClass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class GraphConvClass(ExtendedNNModule):
	def __init__(self,
					heteroData: HeteroData,
					hidden_channels=BTNHGV2ParameterClass.hidden_channels,
					out_channels=BTNHGV2ParameterClass.out_channels,
					num_layers=BTNHGV2ParameterClass.num_layers,
					num_heads=BTNHGV2ParameterClass.num_heads,  # GraphConv 不用 heads，但保留参数以兼容
					dropout=BTNHGV2ParameterClass.dropout):
		super().__init__()
		self._dropout = nn.Dropout(p=dropout)
		# self.heteroDataCls = heteroDataCls
		self.heteroData = heteroData
		self.num_heads = num_heads
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.num_layers = num_layers

		# 前 num_layers-1 层: HeteroConv(GraphConv)
		self.convs = nn.ModuleList()
		self.norm_dicts = nn.ModuleList()

		# 初始化每个节点类型的输入维度
		in_dims = {ntype: self.heteroData[ntype].x.size(-1) for ntype in self.heteroData.node_types}

		for i in range(num_layers - 1):
			conv_dict = {}
			for edge_type in self.heteroData.edge_types:
				src, rel, dst = edge_type
				src_dim = in_dims[src]  # 当前层源节点的输入维度
				conv_dict[edge_type] = GraphConv(in_channels=-1,
												out_channels=hidden_channels)
			self.convs.append(HeteroConv(conv_dict, aggr='sum'))

			# 每层对所有节点类型做 BatchNorm
			norm_dict = nn.ModuleDict()
			for node_type in self.heteroData.node_types:
				norm_dict[node_type] = nn.BatchNorm1d(hidden_channels)
				in_dims[node_type] = hidden_channels  # 更新该节点类型的维度
			self.norm_dicts.append(norm_dict)

		# 最后一层: 针对 "address" 节点的分类
		self.final_conv = GraphConv(
			in_channels=in_dims["address"],  # 此时 address 节点的维度已经是 hidden_channels
			out_channels=out_channels
		)

	def forward(self, hetero_data):
		x_dict, edge_index_dict = hetero_data.x_dict, hetero_data.edge_index_dict

		# 前几层异构卷积 + BN + ReLU + Dropout
		for i, conv in enumerate(self.convs):
			x_dict = conv(x_dict, edge_index_dict)
			new_x_dict = {}
			for key, x in x_dict.items():
				x = self.norm_dicts[i][key](x)   # 每个节点类型单独 BN
				x = F.relu(x)
				x = self._dropout(x)
				new_x_dict[key] = x
			x_dict = new_x_dict

		# 最后一层只对 "address" 节点分类
		if ("coin","addr_to_coin_rev","address") not in edge_index_dict:
			raise ValueError("未找到 ('coin','addr_to_coin_rev','address') 边，请检查 heteroData.edge_types")

		address_x = x_dict["address"]
		coin_x = x_dict["coin"]
		edge_index = edge_index_dict[("coin","addr_to_coin_rev","address")]

		out = self.final_conv((coin_x, address_x), edge_index)
		return out  # shape: [num_address_nodes, out_channels]



