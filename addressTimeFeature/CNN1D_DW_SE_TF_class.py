from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ExtendedNNModule import ExtendedNNModule
from torch.utils.data import TensorDataset, DataLoader
from nnModule import SEBlock

class CNN1D_DW_SE_TF_class(ExtendedNNModule):
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

		self.dropout = nn.Dropout(dropout_rate)

		# Depthwise Convolution 通道卷积
		self.convDW = nn.Conv1d(C, C, cnn_kernel_height, padding="same", groups=C)

		self.bnDW = nn.BatchNorm1d(C)

		# SE 通道注意力
		self.se = SEBlock(C, reduction=4)
		
		self.ln = nn.LayerNorm(C)		

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
		# [B, T, D] → [B, D, T]
		x = x.permute(0, 2, 1)

		# DW
		x = self.convDW(x)
		x = self.bnDW(x)
		x = F.relu(x)
		x = self.dropout(x)

		# SE 通道注意力
		x = self.se(x)

		# [B, D, T] → [B, T, D]		
		x = x.permute(0, 2, 1)
		x = self.ln(x)

		# Transformer
		x = self.transformer(x)

		# Flatten + FC
		x = torch.flatten(x, 1)
		x = self.fc_out(self.dropout(x))

		return x


# # -------------------------
# # SE 通道注意力模块
# # -------------------------
# class SEBlock(nn.Module):
# 	def __init__(self, channels, reduction=4):
# 		super().__init__()
# 		hidden = max(1, channels // reduction)

# 		self.avg_pool = nn.AdaptiveAvgPool1d(1)  # [B, C, T] → [B, C, 1]
# 		self.fc = nn.Sequential(
# 			nn.Linear(channels, hidden),
# 			nn.ReLU(inplace=True),
# 			nn.Linear(hidden, channels),
# 			nn.Sigmoid()
# 		)

# 	def forward(self, x):
# 		b, c, t = x.size()  # [B, C, T]

# 		y = self.avg_pool(x).view(b, c)  # Squeeze：[B, C, T] → [B, C]

# 		y = self.fc(y)  # Excitation：[B, C] → [B, C]（学习通道权重）

# 		y = y.view(b, c, 1)  # 重塑维度：[B, C] → [B, C, 1]

# 		return x * y  # Scale：将通道权重应用到原始特征图


