from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ExtendedNNModule import ExtendedNNModule
from torch.utils.data import TensorDataset, DataLoader

class CNN1D_DW_PW_class(nn.Module):
    """
    Depthwise + Pointwise 1D CNN（MobileNet 风格）
    输入: [batch, seq_len, feature_dim]
    转换: [batch, feature_dim, seq_len]
    """

    def __init__(self,
                 addressTimeFeature_dataSet,
                 cnn_kernel_height=BTNHGV2ParameterClass.cnn_kernel_height,
                 dropout_rate=BTNHGV2ParameterClass.dropout):

        super().__init__()

        self.feature_dim = addressTimeFeature_dataSet.tensors[0].shape[-1]  # D
        self.seq_len = addressTimeFeature_dataSet.tensors[0].shape[-2]      # T=64
        self.num_classes = addressTimeFeature_dataSet.tensors[1].unique().numel()

        self.cnn_in_channels = self.feature_dim

        # -------------------------
        # Depthwise Conv（每通道独立卷积）
        # -------------------------
        self.dw = nn.Conv1d(
            in_channels=self.cnn_in_channels,
            out_channels=self.cnn_in_channels,
            kernel_size=cnn_kernel_height,
            padding="same",
            groups=self.cnn_in_channels
        )
        self.bn_dw = nn.BatchNorm1d(self.cnn_in_channels)

        # -------------------------
        # Pointwise Conv（通道融合）
        # -------------------------
        self.pw = nn.Conv1d(
            in_channels=self.cnn_in_channels,
            out_channels=self.cnn_in_channels,  # 也可以改成更大的通道数
            kernel_size=1
        )
        self.bn_pw = nn.BatchNorm1d(self.cnn_in_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Flatten 后维度
        flattened_size = self.cnn_in_channels * self.seq_len

        # 输出层
        self.fc_out = nn.Linear(flattened_size, self.num_classes)

    def forward(self, x):
        # 输入: [B, T, D] → 转为 [B, D, T]
        x = x.permute(0, 2, 1)

        # Depthwise
        x = F.relu(self.bn_dw(self.dw(x)))

        # Pointwise
        x = F.relu(self.bn_pw(self.pw(x)))

        # Flatten + FC
        x = torch.flatten(x, 1)
        x = self.fc_out(self.dropout(x))

        return x
