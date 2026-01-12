import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
# from torch_geometric.transforms import ToHomogeneous
# from torch_geometric.utils import to_homogeneous
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from ExtendedNNModule import ExtendedNNModule

class RGCNClass(ExtendedNNModule):
	"""
	RGCN 异构图分类器 —— 只对 'address' 节点中 train_mask==True 的节点进行分类
	"""
	def __init__(self,
				heteroData: HeteroData,
				hidden_channels=BTNHGV2ParameterClass.hidden_channels,
				out_channels=BTNHGV2ParameterClass.out_channels,				
				num_layers=BTNHGV2ParameterClass.num_layers,
				dropout=BTNHGV2ParameterClass.dropout):		
		super().__init__()
		# self.heteroDataCls = heteroDataCls
		self.heteroData = heteroData
		self._num_layers = num_layers
		self._node_types = list(self.heteroData.node_types)
		self._dropout = nn.Dropout(p=dropout)

		self._in_proj = nn.ModuleDict()
		for node_type in self._node_types:
			in_channels = self.heteroData[node_type].num_features
			self._in_proj[node_type] = nn.Linear(in_channels, hidden_channels)
		# self._in_proj = nn.ModuleDict({
		# 	ntype: nn.Linear(self.heteroData[ntype].x.shape[1], hidden_channels)
		# 	for ntype in self._node_types
		# })  两种self._in_proj写法等价
		self._num_relations = len(self.heteroData.edge_types)
		self.convs = nn.ModuleList()
		self.btnorms = nn.ModuleList()
		for _ in range(self._num_layers):
			self.convs.append(RGCNConv(hidden_channels, hidden_channels,
						num_relations=self._num_relations if self._num_relations else 0))
			self.btnorms.append(nn.BatchNorm1d(hidden_channels))
		
		self._classifier = nn.Linear(hidden_channels, out_channels)

	def forward(self, data: HeteroData) -> torch.Tensor:
		# 1) 投影各类型节点特征
		for ntype in self._node_types:
			if data[ntype].x is None:
				raise ValueError(f"Missing .x for node type '{ntype}'")
			data[ntype].x = self._in_proj[ntype](data[ntype].x)
			# data[ntype].x = F.relu(data[ntype].x)

		# 2) 转为 homogeneous 图
		data_h = data.to_homogeneous()
		x, edge_index = data_h.x, data_h.edge_index
		edge_type = data_h.edge_type
		node_type = data_h.node_type 

		# 3) RGCN 层
		for conv, norm in zip(self.convs, self.btnorms):
			x = conv(x, edge_index, edge_type)
			x = norm(x)
			x = F.relu(x)
			x = self._dropout(x)

		# 4) 只取 address 节点的表示
		address_type_id = data.node_types.index("address")   # 从原始 hetero 图获取
		address_mask = (node_type == address_type_id)

		# 5) 取出所有 address 节点（不再筛选 train_mask）
		address_x = x[address_mask]                 # homogeneous 中的所有 address 节点

		# 6) 分类
		# address_x = self._dropout(address_x)
		out = self._classifier(address_x)
		return out
