import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData

class HGTClassifier(nn.Module):
    def __init__(
        self,
        metadata,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        heads=4,
        dropout=0.2,
        proj=True,
        target_ntype="address",   # 指定分类的目标节点类型
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.target_ntype = target_ntype

        # 输入投影
        if proj:
            self.proj = nn.ModuleDict({
                ntype: nn.Linear(in_channels[ntype], hidden_channels)
                for ntype in node_types
            })
        else:
            self.proj = nn.ModuleDict()

        # HGT 层
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=heads,
                group='sum'
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_channels) for ntype in node_types
        })

        # 分类头：只针对目标节点类型
        self.cls = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        # 1) 输入投影
        h = {}
        for ntype in data.node_types:
            x = data[ntype].x
            if ntype in self.proj:
                h[ntype] = self.proj[ntype](x)
            else:
                h[ntype] = x
            h[ntype] = F.dropout(h[ntype], p=self.dropout, training=self.training)

        # 2) HGT 层传播
        for conv in self.convs:
            h = conv(h, data.edge_index_dict)
            for ntype in h:
                h[ntype] = self.norms[ntype](h[ntype])
                h[ntype] = F.relu(h[ntype])
                h[ntype] = F.dropout(h[ntype], p=self.dropout, training=self.training)

        # 3) 只取目标节点类型的嵌入
        target_h = h[self.target_ntype]   # [num_address_nodes, hidden]

        # 4) 分类头
        logits = self.cls(target_h)       # [num_address_nodes, out_channels]

        return logits
