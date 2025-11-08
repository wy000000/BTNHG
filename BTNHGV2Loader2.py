import time
print("start import")
time1 = time.time()
import pandas as pd
import torch
from torch_geometric.data import HeteroData
time2 = time.time()
print("import time: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

print("start construct HeteroData")
time1 = time.time()
path = r"D:\BTNHG\BTNHGV2"

# 1. 读取三个特征文件和一个边文件
addr_feat_df = pd.read_csv(path + "/addressFeature.csv")
coin_feat_df = pd.read_csv(path + "/coinFeature.csv")
tx_feat_df   = pd.read_csv(path + "/TxFeature.csv")
edge_df      = pd.read_csv(path + "/hgEdgeV2.csv")

# 2. 建立 ID 映射（保证 PyG 节点索引连续）
address_ids = addr_feat_df['addressID'].unique()
coin_ids    = coin_feat_df['coinID'].unique()
tx_ids      = tx_feat_df['TxID'].unique()

address_id_map = {id_: i for i, id_ in enumerate(address_ids)}
coin_id_map    = {id_: i for i, id_ in enumerate(coin_ids)}
tx_id_map      = {id_: i for i, id_ in enumerate(tx_ids)}

# 3. 初始化 HeteroData
data = HeteroData()

# 4. 构建节点特征矩阵（向量化）
# 映射并排序，保证张量行号和索引一致
addr_feat_df['mapped_id'] = addr_feat_df['addressID'].map(address_id_map)
addr_feat_df_sorted = addr_feat_df.sort_values('mapped_id')
address_x = torch.tensor(
    addr_feat_df_sorted.drop(columns=['addressID','mapped_id']).values,
    dtype=torch.float
)

coin_feat_df['mapped_id'] = coin_feat_df['coinID'].map(coin_id_map)
coin_feat_df_sorted = coin_feat_df.sort_values('mapped_id')
coin_x = torch.tensor(
    coin_feat_df_sorted.drop(columns=['coinID','mapped_id']).values,
    dtype=torch.float
)

tx_feat_df['mapped_id'] = tx_feat_df['TxID'].map(tx_id_map)
tx_feat_df_sorted = tx_feat_df.sort_values('mapped_id')
tx_x = torch.tensor(
    tx_feat_df_sorted.drop(columns=['TxID','mapped_id']).values,
    dtype=torch.float
)

# 挂到 HeteroData
data['address'].x = address_x
data['coin'].x    = coin_x
data['tx'].x      = tx_x

# 5. 建立边关系（向量化）
# addressID → coinID
edge_df['src_address'] = edge_df['addressID'].map(address_id_map)
edge_df['dst_coin']    = edge_df['coinID'].map(coin_id_map)
mask1 = edge_df['src_address'].notna() & edge_df['dst_coin'].notna()
data['address', 'addr_to_coin', 'coin'].edge_index = torch.tensor(
    [edge_df.loc[mask1, 'src_address'].astype(int).values,
     edge_df.loc[mask1, 'dst_coin'].astype(int).values], dtype=torch.long)

# txID_coin → coinID
edge_df['src_tx']   = edge_df['txID_coin'].map(tx_id_map)
edge_df['dst_coin2'] = edge_df['coinID'].map(coin_id_map)
mask2 = edge_df['src_tx'].notna() & edge_df['dst_coin2'].notna()
data['tx', 'tx_to_coin', 'coin'].edge_index = torch.tensor(
    [edge_df.loc[mask2, 'src_tx'].astype(int).values,
     edge_df.loc[mask2, 'dst_coin2'].astype(int).values], dtype=torch.long)

# coinID → coin_txID
edge_df['src_coin3'] = edge_df['coinID'].map(coin_id_map)
edge_df['dst_tx']    = edge_df['coin_txID'].map(tx_id_map)
mask3 = edge_df['src_coin3'].notna() & edge_df['dst_tx'].notna()
data['coin', 'coin_to_tx', 'tx'].edge_index = torch.tensor(
    [edge_df.loc[mask3, 'src_coin3'].astype(int).values,
     edge_df.loc[mask3, 'dst_tx'].astype(int).values], dtype=torch.long)

# 6. 给 address 节点加上 clusterID 标签（如果存在）
if 'clusterID' in addr_feat_df.columns:
    address_y = torch.tensor(
        addr_feat_df_sorted['clusterID'].fillna(-1).astype(int).values,
        dtype=torch.long
    )
    data['address'].y = address_y
else:
    # 如果 clusterID 在 edge_df 中
    address_y = torch.full((len(address_id_map),), -1, dtype=torch.long)
    mask4 = edge_df['addressID'].map(address_id_map).notna() & edge_df['clusterID'].notna()
    for addr_id, cluster in zip(edge_df.loc[mask4, 'addressID'], edge_df.loc[mask4, 'clusterID']):
        idx = address_id_map[addr_id]
        address_y[idx] = int(cluster)
    data['address'].y = address_y

time2 = time.time()
print("construct HeteroData time: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
