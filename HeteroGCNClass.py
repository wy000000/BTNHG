import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, HeteroConv

from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from ExtendedNNModule import ExtendedNNModule


class HeteroGCNClass(ExtendedNNModule):
	def __init__(
		self,
		heteroData: HeteroData,
		hidden_channels=BTNHGV2ParameterClass.hidden_channels,
		num_layers=BTNHGV2ParameterClass.num_layers,
		dropout=BTNHGV2ParameterClass.dropout,
	):
		super().__init__()
		self.heteroData = heteroData
		self.hidden_channels = hidden_channels
		self.num_layers = num_layers
		self._dropout = nn.Dropout(p=dropout)

		# address 节点的有效标签数即分类数，忽略 -1 的未标注节点。
		valid_y = self.heteroData["address"].y
		valid_y = valid_y[valid_y != -1]
		self.num_classes = int(valid_y.unique().numel())

		self.input_proj = nn.ModuleDict({
			node_type: nn.Linear(
				self.heteroData[node_type].x.size(-1),
				self.hidden_channels,
			)
			for node_type in self.heteroData.node_types
		})

		self.convs = nn.ModuleList()
		self.norms = nn.ModuleList()

		for _ in range(self.num_layers):
			conv_dict = {}
			for edge_type in self.heteroData.edge_types:
				# GraphConv 支持二部图消息传递，适合当前 address/coin/tx 异构边。
				conv_dict[edge_type] = GraphConv(
					in_channels=self.hidden_channels,
					out_channels=self.hidden_channels,
				)
			self.convs.append(HeteroConv(conv_dict, aggr="sum"))
			self.norms.append(nn.ModuleDict({
				node_type: nn.LayerNorm(self.hidden_channels)
				for node_type in self.heteroData.node_types
			}))

		self.classifier = nn.Sequential(
			nn.Linear(self.hidden_channels, self.hidden_channels),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.Linear(self.hidden_channels, self.num_classes),
		)

	def forward(self, hetero_data: HeteroData) -> torch.Tensor:
		x_dict = {
			node_type: self.input_proj[node_type](hetero_data[node_type].x)
			for node_type in hetero_data.node_types
		}
		edge_index_dict = hetero_data.edge_index_dict

		for conv, norm_dict in zip(self.convs, self.norms):
			h_dict = conv(x_dict, edge_index_dict)
			new_x_dict = {}
			for node_type, h in h_dict.items():
				# 残差连接有助于让浅层图卷积更稳定。
				h = h + x_dict[node_type]
				h = norm_dict[node_type](h)
				h = F.relu(h)
				h = self._dropout(h)
				new_x_dict[node_type] = h
			x_dict = new_x_dict

		return self.classifier(x_dict["address"])
