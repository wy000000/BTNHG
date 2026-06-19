import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv,
    GATConv,
    SAGEConv,
    TransformerConv
)
from torch_geometric.data import HeteroData
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from ExtendedNNModule import ExtendedNNModule


class GraphGPSClass(ExtendedNNModule):
    """
    GraphGPS (Graph Position-aware Graph Transformer) 异构图分类器
    只对 'address' 节点进行分类
    """

    def __init__(self,
                 heteroData: HeteroData,
                 hidden_channels=BTNHGV2ParameterClass.hidden_channels,
                 out_channels=BTNHGV2ParameterClass.out_channels,
                 num_heads=BTNHGV2ParameterClass.num_heads,
                 num_layers=BTNHGV2ParameterClass.num_layers,
                 dropout=BTNHGV2ParameterClass.dropout):
        super().__init__()
        self.heteroData = heteroData
        self._metadata = self.heteroData.metadata()
        self._node_types, self._edge_types = self._metadata
        self._num_classes = self.heteroData["address"].y.unique().numel() - 1
        self._hidden_channels = hidden_channels
        self._out_channels = out_channels
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout = nn.Dropout(p=dropout)

        # 输入投影层：将不同节点类型的特征投影到相同维度
        self._in_proj = nn.ModuleDict()
        for ntype in self._node_types:
            in_channels = self.heteroData[ntype].x.shape[1]
            self._in_proj[ntype] = nn.Linear(in_channels, hidden_channels)

        # 构建多层异构卷积 (局部消息传递)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            conv_dict = {}
            for edge_type in self._edge_types:
                src, rel, dst = edge_type
                in_src = hidden_channels
                in_dst = hidden_channels

                # 使用SAGEConv作为基础卷积（不需要边特征）
                conv_dict[edge_type] = SAGEConv(
                    in_channels=(in_src, in_dst),
                    out_channels=hidden_channels,
                    aggr='mean'
                )

            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

            # 每层对所有节点类型做LayerNorm
            norm_dict = nn.ModuleDict()
            for node_type in self._node_types:
                norm_dict[node_type] = nn.LayerNorm(hidden_channels)
            self.norms.append(norm_dict)

        # Transformer层（模拟GPS的全局注意力）
        self.transformer_convs = nn.ModuleList()
        for _ in range(num_layers):
            transformer_dict = {}
            for edge_type in self._edge_types:
                src, rel, dst = edge_type
                transformer_dict[edge_type] = TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout
                )
            self.transformer_convs.append(HeteroConv(transformer_dict, aggr='sum'))

        # 分类头：只针对address节点
        self.cls = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            self._dropout,
            nn.Linear(hidden_channels, self._num_classes)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        # 1) 输入投影
        x_dict = {}
        for ntype in data.node_types:
            x = data[ntype].x
            x_dict[ntype] = self._in_proj[ntype](x)

        # 2) 多层传播 (局部GNN + Transformer注意力)
        edge_index_dict = data.edge_index_dict

        for i, (conv, norm_dict, transformer_conv) in enumerate(zip(self.convs, self.norms, self.transformer_convs)):
            # 局部消息传递（SAGE）
            x_local = conv(x_dict, edge_index_dict)

            # 全局注意力（Transformer）
            x_global = transformer_conv(x_dict, edge_index_dict)

            # 融合局部和全局信息
            new_x_dict = {}
            for ntype in x_local:
                # 残差连接
                x = x_local[ntype] + x_global[ntype] + x_dict[ntype]
                # 归一化
                x = norm_dict[ntype](x)
                # 激活
                x = F.relu(x)
                # Dropout
                x = self._dropout(x)
                new_x_dict[ntype] = x

            x_dict = new_x_dict

        # 3) 只取address节点的嵌入
        target_h = x_dict["address"]

        # 4) 分类头
        logits = self.cls(target_h)

        return logits