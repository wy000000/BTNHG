import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
from ExtendedNNModule import ExtendedNNModule

class HGTClass(ExtendedNNModule):
	def __init__(self,
				heteroDataCls: BTNHGV2HeteroDataClass,
				hidden_channels=BTNHGV2ParameterClass.hidden_channels,
				out_channels=BTNHGV2ParameterClass.out_channels,
				num_heads=BTNHGV2ParameterClass.num_heads,
				num_layers=BTNHGV2ParameterClass.num_layers,
				dropout=BTNHGV2ParameterClass.dropout,
				useProj=BTNHGV2ParameterClass.HGT_useProj,
				batch_size=BTNHGV2ParameterClass.batch_size,
				shuffle=BTNHGV2ParameterClass.shuffle,
				resetSeed=BTNHGV2ParameterClass.resetSeed):
		super().__init__()
		self.heteroDataCls = heteroDataCls
		self.heteroDataCls.getTrainTestMask()
		self.heteroData = heteroDataCls.heteroData
		self._metadata = self.heteroData.metadata()		
		self._out_channels = out_channels
		self._node_types, self._edge_types = self._metadata
		self._hidden_channels = hidden_channels
		self.batch_size=batch_size
		self.shuffle=shuffle
		self.resetSeed=resetSeed
		self._dropout = nn.Dropout(p=dropout)
		self._num_heads = num_heads
		self._num_layers = num_layers

		self._useProj = useProj
		self._proj = None
		# 提取每种节点类型的输入维度
		self._in_channels = {
			ntype: self.heteroData[ntype].x.shape[1]
			for ntype in self.heteroData.node_types}

		# 输入投影
		if self._useProj:
			self._proj = nn.ModuleDict({
				ntype: nn.Linear(self._in_channels[ntype], self._hidden_channels)
				for ntype in self._node_types})
		else:
			self._proj = nn.ModuleDict()

		# HGT层
		self.convs = nn.ModuleList([
						HGTConv(
							in_channels=self._hidden_channels,
							out_channels=self._hidden_channels,
							metadata=self._metadata,
							heads=self._num_heads)
						for _ in range(self._num_layers)
					])

		self.norms = nn.ModuleDict({
			ntype: nn.LayerNorm(self._hidden_channels) for ntype in self._node_types})

		# 分类头：只针对目标节点类型
		self.cls = nn.Sequential(
			# nn.Linear(self._hidden_channels, self._hidden_channels),
			# nn.ReLU(),
			# self._dropout,
			nn.Linear(self._hidden_channels, self._out_channels)
		)

	def forward(self, data: HeteroData) -> torch.Tensor:
		# 1) 输入投影
		h = {}
		for ntype in data.node_types:
			x = data[ntype].x
			if ntype in self._proj:
				h[ntype] = self._proj[ntype](x)
			else:
				h[ntype] = x
			# h[ntype] = self._dropout(h[ntype])   # 输入投影后可选 Dropout

		# 2) HGT 层传播 (Conv → Residual → Norm → ReLU → Dropout)
		for conv in self.convs:
			h_new = conv(h, data.edge_index_dict)   # HGTConv 输出
			for ntype in h_new:
				# 残差连接
				h_new[ntype] = h_new[ntype] + h[ntype]

				# 归一化 (推荐 LayerNorm)
				h_new[ntype] = self.norms[ntype](h_new[ntype])

				# 激活函数
				h_new[ntype] = F.relu(h_new[ntype])

				# Dropout
				h_new[ntype] = self._dropout(h_new[ntype])

			h = h_new   # 更新 h

		# 3) 只取目标节点类型的嵌入
		target_h = h["address"]   # [num_address_nodes, hidden]

		# 4) 分类头
		logits = self.cls(target_h)   # [num_address_nodes, out_channels]

		return logits

