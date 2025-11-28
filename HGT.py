import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean, scatter_sum, scatter_max

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
        readout="mean",
        proj=True,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.dropout = dropout

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

        self.readout_mode = readout
        self.cls = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        输入: HeteroData
        输出: 图级分类 logits
        """
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

        # 3) 读出为图级表示
        pooled = []
        for ntype in h:
            if h[ntype].numel() > 0:
                if hasattr(data[ntype], "batch") and data[ntype].batch is not None:
                    batch = data[ntype].batch
                    pooled.append(scatter_mean(h[ntype], batch, dim=0))
                else:
                    pooled.append(h[ntype].mean(dim=0, keepdim=True))
        g = torch.stack(pooled, dim=0).mean(dim=0)

        # 4) 分类头
        logits = self.cls(g)
        return logits
