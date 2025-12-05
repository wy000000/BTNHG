import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, GraphConv
from ExtendedNNModule import ExtendedNNModule
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
from BTNHGV2ParameterClass import BTNHGV2ParameterClass

class GraphConvClass(ExtendedNNModule):
	def __init__(self,
				heteroDataCls: BTNHGV2HeteroDataClass,
				hidden_channels=BTNHGV2ParameterClass.hidden_channels,
				out_channels=BTNHGV2ParameterClass.out_channels,
				num_layers=BTNHGV2ParameterClass.num_layers,
				num_heads=BTNHGV2ParameterClass.num_heads,
				batch_size=BTNHGV2ParameterClass.batch_size,
				shuffle=BTNHGV2ParameterClass.shuffle,
				resetSeed=BTNHGV2ParameterClass.resetSeed,
				dropout=BTNHGV2ParameterClass.dropout):
		super(GraphConvClass, self).__init__()
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.resetSeed = resetSeed
		self._dropout = nn.Dropout(p=dropout)
		self.heteroDataCls = heteroDataCls
		self.heteroDataCls.getTrainTestMask()
		self.heteroData = heteroDataCls.heteroData
		self.num_heads = num_heads
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.num_layers = num_layers



	def forward(self, hetero_data):

