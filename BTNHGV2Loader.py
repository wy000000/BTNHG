print("start import")
import time
time1 = time.time()
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
import numpy as np
time2 = time.time()
print("import time: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

print("start construct HeteroData")
time1 = time.time()
path = r"D:\BTNHG\BTNHGV2"

# 1. 读取数据
print("start read data")
time1 = time.time()
addr_feat_df = pd.read_csv(os.path.join(path, "addressFeature.csv"))
coin_feat_df = pd.read_csv(os.path.join(path, "coinFeature.csv"))
tx_feat_df   = pd.read_csv(os.path.join(path, "TxFeature.csv"))
edge_df      = pd.read_csv(os.path.join(path, "hgEdgeV2.csv"))
time2 = time.time()
print(f"读取数据时间: {time2 - time1}")
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

# 2. 建立 ID 映射
print("construct ID map and 构建节点特征矩阵")
time1 = time.time()
address_ids = addr_feat_df['addressID'].unique()
coin_ids    = coin_feat_df['coinID'].unique()
tx_ids      = tx_feat_df['txID'].unique()

address_id_map = {id_: i for i, id_ in enumerate(address_ids)}
coin_id_map    = {id_: i for i, id_ in enumerate(coin_ids)}
tx_id_map      = {id_: i for i, id_ in enumerate(tx_ids)}

# 3. 初始化 HeteroData
data = HeteroData()

# 4. 构建节点特征矩阵（直接用 replace + values）
data['address'].x = torch.tensor(
    addr_feat_df.drop(columns=['addressID']).values, dtype=torch.float
)
data['coin'].x = torch.tensor(
    coin_feat_df.drop(columns=['coinID']).values, dtype=torch.float
)
data['tx'].x = torch.tensor(
    tx_feat_df.drop(columns=['txID']).values, dtype=torch.float
)
time2 = time.time()
print(f"构建ID map and 节点特征矩阵时间: {time2 - time1}")
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

# 5. 建立边关系（向量化处理）
print("构建边关系")
time1 = time.time()
def build_edge(df, src_col, dst_col, src_map, dst_map):
    """通用边构造函数"""
    src = df[src_col].map(src_map)
    dst = df[dst_col].map(dst_map)
    valid = src.notna() & dst.notna()
    edge_index = np.column_stack([src[valid].astype(int).to_numpy(),
                                  dst[valid].astype(int).to_numpy()]).T
    return torch.from_numpy(edge_index).long()

# 构建三类边
data['address', 'addr_to_coin', 'coin'].edge_index = build_edge(edge_df, 'addressID', 'coinID',
                                                                address_id_map, coin_id_map)

data['tx', 'tx_to_coin', 'coin'].edge_index = build_edge(edge_df, 'txID_coin', 'coinID',
                                                         tx_id_map, coin_id_map)

data['coin', 'coin_to_tx', 'tx'].edge_index = build_edge(edge_df, 'coinID', 'coin_txID',
                                                         coin_id_map, tx_id_map)
time2 = time.time()
print(f"构建边关系时间: {time2 - time1}")
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

# 6. 给 address 节点加标签
print("给 address 节点加标签")
time1 = time.time()
address_y = torch.full((len(address_id_map),), -1, dtype=torch.long)
valid_cluster = edge_df[['addressID', 'clusterID']].dropna()
for addr, cluster in zip(valid_cluster['addressID'], valid_cluster['clusterID']):
    if addr in address_id_map:
        address_y[address_id_map[addr]] = int(cluster)
data['address'].y = address_y

time2 = time.time()
print("给 address 节点加标签时间: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")