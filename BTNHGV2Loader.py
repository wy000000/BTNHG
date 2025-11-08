import pandas as pd
import torch
from torch_geometric.data import HeteroData

# 1. 读取三个特征文件和一个边文件
addr_feat_df = pd.read_csv("addressFeature.csv")
coin_feat_df = pd.read_csv("coinFeature.csv")
tx_feat_df   = pd.read_csv("TxFeature.csv")
edge_df = pd.read_csv("hgEdgeV2.csv")

# 2. 建立 ID 映射（保证 PyG 节点索引连续）
address_ids = addr_feat_df['addressID'].unique()
coin_ids    = coin_feat_df['coinID'].unique()
tx_ids      = tx_feat_df['TxID'].unique()   # 注意这里改成 TxID

address_id_map = {id_: i for i, id_ in enumerate(address_ids)}
coin_id_map    = {id_: i for i, id_ in enumerate(coin_ids)}
tx_id_map      = {id_: i for i, id_ in enumerate(tx_ids)}

# 3. 初始化 HeteroData
data = HeteroData()

# 4. 构建节点特征矩阵
# 去掉 ID 列，只保留数值特征
addr_features = addr_feat_df.drop(columns=['addressID']).values.astype(float)
coin_features = coin_feat_df.drop(columns=['coinID']).values.astype(float)
tx_features   = tx_feat_df.drop(columns=['TxID']).values.astype(float)

# 转换为 torch.tensor，并对齐索引
address_x = torch.zeros((len(address_ids), addr_features.shape[1]))
for _, row in addr_feat_df.iterrows():
    idx = address_id_map[row['addressID']]
    address_x[idx] = torch.tensor(row.drop(labels=['addressID']).values, dtype=torch.float)

coin_x = torch.zeros((len(coin_ids), coin_features.shape[1]))
for _, row in coin_feat_df.iterrows():
    idx = coin_id_map[row['coinID']]
    coin_x[idx] = torch.tensor(row.drop(labels=['coinID']).values, dtype=torch.float)

tx_x = torch.zeros((len(tx_ids), tx_features.shape[1]))
for _, row in tx_feat_df.iterrows():
    idx = tx_id_map[row['TxID']]   # 注意这里改成 TxID
    tx_x[idx] = torch.tensor(row.drop(labels=['TxID']).values, dtype=torch.float)

# 5. 挂到 HeteroData，节点类型改为 address、coin、tx
data['address'].x = address_x
data['coin'].x    = coin_x
data['tx'].x      = tx_x


# 之前已经构建了 HeteroData 并载入了节点特征
# data = HeteroData()
# 已有 address_id_map, coin_id_map, tx_id_map

# 6. 建立边关系
# addressID → coinID
src_address = []
dst_coin = []
for _, row in edge_df.iterrows():
    if row['addressID'] in address_id_map and row['coinID'] in coin_id_map:
        src_address.append(address_id_map[row['addressID']])
        dst_coin.append(coin_id_map[row['coinID']])
data['address', 'addr_to_coin', 'coin'].edge_index = torch.tensor([src_address, dst_coin], dtype=torch.long)

# txID_coin → coinID
src_tx = []
dst_coin2 = []
for _, row in edge_df.iterrows():
    if row['txID_coin'] in tx_id_map and row['coinID'] in coin_id_map:
        src_tx.append(tx_id_map[row['txID_coin']])
        dst_coin2.append(coin_id_map[row['coinID']])
data['tx', 'tx_to_coin', 'coin'].edge_index = torch.tensor([src_tx, dst_coin2], dtype=torch.long)

# coinID → coin_txID (注意 coin_txID 也是 TxID，需要用 tx_id_map)
src_coin3 = []
dst_tx = []
for _, row in edge_df.iterrows():
    if row['coinID'] in coin_id_map and row['coin_txID'] in tx_id_map:
        src_coin3.append(coin_id_map[row['coinID']])
        dst_tx.append(tx_id_map[row['coin_txID']])
data['coin', 'coin_to_tx', 'tx'].edge_index = torch.tensor([src_coin3, dst_tx], dtype=torch.long)

# 7. 给 address 节点加上 clusterID 标签（如果存在）
# 初始化标签张量，默认 -1 表示无标签
address_y = torch.full((len(address_id_map),), -1, dtype=torch.long)
for _, row in edge_df.iterrows():
    if row['addressID'] in address_id_map and not pd.isna(row['clusterID']):
        idx = address_id_map[row['addressID']]
        address_y[idx] = int(row['clusterID'])
data['address'].y = address_y



