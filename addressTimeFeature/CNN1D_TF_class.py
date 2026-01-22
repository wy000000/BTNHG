from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ExtendedNNModule import ExtendedNNModule
from torch.utils.data import TensorDataset, DataLoader

#Input → DW → BN → ReLU → SE → (permute) → Transformer → FC
#DW → SE → Transformer
#DW+SE 分支：局部特征, Transformer 分支：全局特征, 最后 concat

class CNN1D_TF_class(nn.Module):
	def __init__(self, addressTimeFeature_dataSet,
				 cnn_kernel_height=BTNHGV2ParameterClass.cnn_kernel_height,
				 dropout_rate=BTNHGV2ParameterClass.dropout,
				 nhead=BTNHGV2ParameterClass.tf_num_heads,
				 num_layers=BTNHGV2ParameterClass.tf_num_layers,
				 dim_feedforward=BTNHGV2ParameterClass.tf_dim_feedforward):

		super().__init__()

		self.feature_dim = addressTimeFeature_dataSet.tensors[0].shape[-1]
		self.seq_len = addressTimeFeature_dataSet.tensors[0].shape[-2]
		self.num_classes = addressTimeFeature_dataSet.tensors[1].unique().numel()

		C = self.feature_dim
		self.input_nn_norm = nn.LayerNorm(C)

		self.dropout = nn.Dropout(dropout_rate)

		# Transformer Encoder
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=C,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout_rate,
			batch_first=True
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		self.fc_out = nn.Linear(C * self.seq_len, self.num_classes)
		
	def forward(self, x):

		# Input Normalization
		x = self.input_nn_norm(x)

		# Transformer
		x = self.transformer(x)

		# Flatten + FC
		x = torch.flatten(x, 1)
		x = self.fc_out(self.dropout(x))

		return x

