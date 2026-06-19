import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from torch_geometric.nn import (
    HeteroConv,
    GINEConv,
    TransformerConv
)
from torch_geometric.data import HeteroData
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from ExtendedNNModule import ExtendedNNModule


class GPSClass(ExtendedNNModule):
    """
    GraphGPS (Graph Position-aware Graph Transformer) 异构图分类器
    使用GINE卷积，支持边特征，只对 'address' 节点进行分类
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

        # 边特征投影层（如果有边特征）- 为每种边类型单独处理
        self._edge_proj = None
        self._has_edge_attr = self._check_has_edge_attr()
        if self._has_edge_attr:
            self._edge_proj = nn.ModuleDict()
            for edge_type in self._edge_types:
                edge_type_str = self._edge_type_to_str(edge_type)
                edge_dim = self._get_edge_dim(edge_type)
                if edge_dim > 0:
                    self._edge_proj[edge_type_str] = nn.Linear(edge_dim, hidden_channels)

        # 构建多层异构卷积 (局部消息传递 - GINE)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            conv_dict = {}
            for edge_type in self._edge_types:
                src, rel, dst = edge_type

                # 使用GINEConv作为基础卷积，支持边特征
                gin_net = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_channels * 2, hidden_channels)
                )
                conv_dict[edge_type] = GINEConv(
                    gin_net,
                    train_eps=True,
                    edge_dim=hidden_channels if self._has_edge_attr else None
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
                    dropout=dropout,
                    edge_dim=hidden_channels if self._has_edge_attr else None
                )
            self.transformer_convs.append(HeteroConv(transformer_dict, aggr='sum'))

        # 分类头：只针对address节点
        self.cls = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            self._dropout,
            nn.Linear(hidden_channels, self._num_classes)
        )

    @staticmethod
    def _edge_type_to_str(edge_type):
        """将边类型元组转换为字符串"""
        return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"

    def _check_has_edge_attr(self):
        """检查是否有边特征"""
        for edge_type in self._edge_types:
            if 'edge_attr' in self.heteroData[edge_type]:
                return True
        return False

    def _get_edge_dim(self, edge_type):
        """获取指定边类型的特征维度"""
        if 'edge_attr' in self.heteroData[edge_type]:
            return self.heteroData[edge_type].edge_attr.shape[1]
        return 0

    def _prepare_edge_attr_dict(self, data):
        """准备边特征字典，投影到统一维度"""
        edge_attr_dict = {}

        if not self._has_edge_attr or self._edge_proj is None:
            return None

        for edge_type in self._edge_types:
            edge_type_str = self._edge_type_to_str(edge_type)
            if 'edge_attr' in data[edge_type] and edge_type_str in self._edge_proj:
                edge_attr = data[edge_type].edge_attr
                # 投影边特征
                edge_attr_dict[edge_type] = self._edge_proj[edge_type_str](edge_attr)
            else:
                edge_attr_dict[edge_type] = None

        return edge_attr_dict

    def forward(self, data: HeteroData) -> torch.Tensor:
        # 1) 输入投影
        x_dict = {}
        for ntype in data.node_types:
            x = data[ntype].x
            x_dict[ntype] = self._in_proj[ntype](x)

        # 2) 准备边特征
        edge_attr_dict = self._prepare_edge_attr_dict(data)
        edge_index_dict = data.edge_index_dict

        # 3) 多层传播 (局部GNN + Transformer注意力)
        for i, (conv, norm_dict, transformer_conv) in enumerate(zip(self.convs, self.norms, self.transformer_convs)):
            # 局部消息传递（GINE + 边特征）
            if edge_attr_dict is not None:
                x_local = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                x_local = conv(x_dict, edge_index_dict)

            # 全局注意力（Transformer）
            if edge_attr_dict is not None:
                x_global = transformer_conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
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

        # 4) 只取address节点的嵌入
        target_h = x_dict["address"]

        # 5) 分类头
        logits = self.cls(target_h)

        return logits


class GPSDataEnhancer:
    """
    增强类：用于为异构图添加完整的边特征
    独立使用，不修改项目其他文件
    """

    @staticmethod
    def enhance_with_edge_features(heteroDataCls, dataPath=None):
        """
        为异构图添加完整的边特征

        Args:
            heteroDataCls: BTNHGV2HeteroDataClass的实例
            dataPath: 数据文件路径，默认从BTNHGV2ParameterClass获取

        Returns:
            添加了边特征的heteroData
        """
        if dataPath is None:
            dataPath = BTNHGV2ParameterClass.dataPath

        heteroData = heteroDataCls.heteroData

        # 读取特征文件
        coin_feat_df = pd.read_csv(os.path.join(dataPath, "coinFeature.csv"))
        tx_feat_df = pd.read_csv(os.path.join(dataPath, "TxFeature.csv"))

        # 创建映射字典
        coin_value_map = coin_feat_df.set_index('coinID')['value'].to_dict()
        tx_feat_map = tx_feat_df.set_index('txID').to_dict('index')

        # 反向映射：从索引到原始ID
        idx_to_coin_id = {v: k for k, v in heteroDataCls._coin_id_map.items()}
        idx_to_tx_id = {v: k for k, v in heteroDataCls._tx_id_map.items()}

        # 处理每种边类型
        for edge_type in list(heteroData.edge_types):
            src, rel, dst = edge_type

            # 跳过反向边，处理完原始边后会自动处理
            if rel.endswith('_rev'):
                continue

            edge_index = heteroData[edge_type].edge_index
            num_edges = edge_index.shape[1]

            edge_features = []

            if rel == 'addr_to_coin':
                # address->coin边：特征 = [coin.value]
                for i in range(num_edges):
                    coin_idx = edge_index[1, i].item()
                    coin_id = idx_to_coin_id.get(coin_idx, -1)
                    coin_value = coin_value_map.get(coin_id, 0.0)
                    # 对coin_value进行log变换，避免数值过大
                    coin_value_log = torch.log(torch.tensor(coin_value) + 1).item()
                    edge_features.append([coin_value_log])

            elif rel == 'tx_to_coin':
                # tx->coin边：特征 = [coin.value, tx.blockID, tx.movedCoin]
                for i in range(num_edges):
                    coin_idx = edge_index[1, i].item()
                    tx_idx = edge_index[0, i].item()

                    coin_id = idx_to_coin_id.get(coin_idx, -1)
                    tx_id = idx_to_tx_id.get(tx_idx, -1)

                    coin_value = coin_value_map.get(coin_id, 0.0)
                    tx_info = tx_feat_map.get(tx_id, {})

                    block_id = tx_info.get('blockID', 0)
                    moved_coin = tx_info.get('movedCoin', 0)

                    # 对数值进行log变换
                    coin_value_log = torch.log(torch.tensor(coin_value) + 1).item()
                    moved_coin_log = torch.log(torch.tensor(moved_coin) + 1).item()

                    edge_features.append([coin_value_log, block_id, moved_coin_log])

            elif rel == 'coin_to_tx':
                # coin->tx边：特征 = [coin.value, tx.blockID, tx.movedCoin]
                for i in range(num_edges):
                    coin_idx = edge_index[0, i].item()
                    tx_idx = edge_index[1, i].item()

                    coin_id = idx_to_coin_id.get(coin_idx, -1)
                    tx_id = idx_to_tx_id.get(tx_idx, -1)

                    coin_value = coin_value_map.get(coin_id, 0.0)
                    tx_info = tx_feat_map.get(tx_id, {})

                    block_id = tx_info.get('blockID', 0)
                    moved_coin = tx_info.get('movedCoin', 0)

                    # 对数值进行log变换
                    coin_value_log = torch.log(torch.tensor(coin_value) + 1).item()
                    moved_coin_log = torch.log(torch.tensor(moved_coin) + 1).item()

                    edge_features.append([coin_value_log, block_id, moved_coin_log])

            # 设置边特征
            if edge_features:
                heteroData[edge_type].edge_attr = torch.tensor(
                    edge_features, dtype=torch.float32
                )

        # 为反向边复制边特征
        GPSDataEnhancer._copy_edge_features_to_reverse(heteroData)

        return heteroData

    @staticmethod
    def _copy_edge_features_to_reverse(heteroData):
        """为反向边复制边特征"""
        for (src, rel, dst) in list(heteroData.edge_types):
            if rel.endswith('_rev'):
                continue

            # 查找对应的反向边
            rel_rev = rel + '_rev'
            reverse_edge_type = (dst, rel_rev, src)

            if reverse_edge_type in heteroData.edge_types:
                # 如果原边有特征，复制给反向边
                if 'edge_attr' in heteroData[(src, rel, dst)]:
                    heteroData[reverse_edge_type].edge_attr = \
                        heteroData[(src, rel, dst)].edge_attr.clone()


def create_gps_model_with_edge_features(heteroDataCls, dataPath=None):
    """
    便捷函数：创建带有完整边特征的GPS模型

    Args:
        heteroDataCls: BTNHGV2HeteroDataClass的实例
        dataPath: 数据文件路径

    Returns:
        GPSClass模型实例
    """
    # 添加边特征
    heteroData = GPSDataEnhancer.enhance_with_edge_features(heteroDataCls, dataPath)

    # 创建模型
    model = GPSClass(heteroData=heteroData)

    return model


# 导出符号，确保可以正确导入
__all__ = ['GPSClass', 'GPSDataEnhancer', 'create_gps_model_with_edge_features']