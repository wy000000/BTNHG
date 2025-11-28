import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import to_homogeneous


class HeteroRGCNClassifier(nn.Module):
    """
    RGCN 异构图分类器 —— 只对 'address' 节点中 train_mask==True 的节点进行分类
    """
    def __init__(self,
                 in_channels_dict: dict,
                 hidden_channels: int,
                 out_channels: int,
                 num_relations: int | None = None,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.node_types = list(in_channels_dict.keys())
        self.in_proj = nn.ModuleDict({
            ntype: nn.Linear(in_channels_dict[ntype], hidden_channels)
            for ntype in self.node_types
        })

        self.num_relations = num_relations
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels,
                                       num_relations=self.num_relations if self.num_relations else 0))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)

        self._initialized_relations = self.num_relations is not None

    def forward(self, data: HeteroData) -> torch.Tensor:
        # 1) 投影各类型节点特征
        for ntype in self.node_types:
            if data[ntype].x is None:
                raise ValueError(f"Missing .x for node type '{ntype}'")
            data[ntype].x = self.in_proj[ntype](data[ntype].x)

        # 2) 转为 homogeneous 图
        data_h = to_homogeneous(data, node_attrs=["x"], edge_attrs=[])
        edge_type = data_h.edge_type

        if not self._initialized_relations:
            num_relations = int(edge_type.max().item()) + 1 if edge_type.numel() > 0 else 0
            self.convs = nn.ModuleList([
                RGCNConv(self.convs[0].in_channels, self.convs[0].out_channels, num_relations=num_relations)
                for _ in range(len(self.convs))
            ])
            self._initialized_relations = True

        x, edge_index = data_h.x, data_h.edge_index

        # 3) RGCN 层
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)

        # 4) 只取 address 节点的表示
        address_type_id = data_h._node_type_dict["address"]
        mask_address = (data_h.node_type == address_type_id)

        # 5) 再筛选 train_mask
        # 注意：to_homogeneous 会把每个节点的 mask 合并成一个 tensor
        # 所以 data_h.train_mask 对应所有节点的 mask
        mask_train = data_h.train_mask & mask_address

        x_address_train = x[mask_train]

        # 6) 分类
        out = self.classifier(x_address_train)
        return out
