# -*- coding: utf-8 -*-

# print("start import")
import time
time1 = time.time()
import os
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
# import sys
time2 = time.time()
# print("import used time: ", time2 - time1)
# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")


class BTNHGV2HeteroDataClass(Dataset):
	"""
	比特币交易网络数据集类，用于处理异构图数据的划分和加载
	"""
	def __init__(self, heteroData=None):
		super().__init__()
		self._heteroData=None
		if heteroData is not None:
			self._heteroData=heteroData	
		else:
			self._loadBTNHGV2Data()
		#添加一个属性，返回_heteroData
	@property
	def heteroData(self):
		return self._heteroData

	def _loadBTNHGV2Data(self, dataPath=BTNHGV2ParameterClass.dataPath):
		print("start construct HeteroData")
		time1 = time.time()

		# 1. 读取数据
		print("start read data")
		time1 = time.time()
		addr_feat_df = pd.read_csv(os.path.join(dataPath, "addressFeature.csv"))
		coin_feat_df = pd.read_csv(os.path.join(dataPath, "coinFeature.csv"))
		tx_feat_df   = pd.read_csv(os.path.join(dataPath, "TxFeature.csv"))
		edge_df      = pd.read_csv(os.path.join(dataPath, "hgEdgeV2.csv"))
		time2 = time.time()
		print(f"读取数据用时: {time2 - time1}")
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
		self._heteroData = HeteroData()

		# 4. 构建节点特征矩阵（直接用 replace + values）
		self._heteroData['address'].x = torch.tensor(
			addr_feat_df.drop(columns=['addressID']).values, dtype=torch.float
		)
		self._heteroData['coin'].x = torch.tensor(
			coin_feat_df.drop(columns=['coinID']).values, dtype=torch.float
		)
		self._heteroData['tx'].x = torch.tensor(
			tx_feat_df.drop(columns=['txID']).values, dtype=torch.float
		)
		time2 = time.time()
		print(f"构建ID map and 节点特征矩阵用时: {time2 - time1}")
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

		# 构建边
		self._heteroData['address', 'addr_to_coin', 'coin'].edge_index = build_edge(edge_df, 'addressID', 'coinID',
																		address_id_map, coin_id_map)

		self._heteroData['tx', 'tx_to_coin', 'coin'].edge_index = build_edge(edge_df, 'txID_coin', 'coinID',
																tx_id_map, coin_id_map)

		self._heteroData['coin', 'coin_to_tx', 'tx'].edge_index = build_edge(edge_df, 'coinID', 'coin_txID',
																coin_id_map, tx_id_map)
		time2 = time.time()
		print(f"构建边关系用时: {time2 - time1}")
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

		# 6. 给 address 节点加标签
		print("给 address 节点加标签")
		time1 = time.time()
		address_y = torch.full((len(address_id_map),), -1, dtype=torch.long)
		# 使用向量化操作
		valid_cluster = edge_df[['addressID', 'clusterID']].dropna()
		# 过滤出有效的地址和聚类ID
		valid_mask = valid_cluster['addressID'].isin(address_id_map.keys())
		valid_data = valid_cluster[valid_mask]
		# 批量转换和赋值
		if not valid_data.empty:
			indices = [address_id_map[addr] for addr in valid_data['addressID']]
			# 这里仍然使用int()转换，但通过列表推导式更高效
			clusters = [int(c) for c in valid_data['clusterID']]
			address_y[indices] = torch.tensor(clusters, dtype=torch.long)
		self._heteroData['address'].y = address_y

		time2 = time.time()
		print("给 address 节点加标签用时: ", time2 - time1)
		
		#输出self.data的所有类型的结点及其特征矩阵的形状
		print("address node types:", self._heteroData.node_types)
		print("address:"+str(self._heteroData['address'].x.shape))
		print("coin:"+str(self._heteroData['coin'].x.shape))
		print("tx:"+str(self._heteroData['tx'].x.shape))
		# 输出self.data的所有类型的边及其边索引的形状
		print("edge types:", self._heteroData.edge_types)
		print("address-coin:"+str(self._heteroData['address', 'addr_to_coin', 'coin'].edge_index.shape))
		print("tx-coin:"+str(self._heteroData['tx', 'tx_to_coin', 'coin'].edge_index.shape))
		print("coin-tx:"+str(self._heteroData['coin', 'coin_to_tx', 'tx'].edge_index.shape))
		#输出self._heteroData.y的形状
		print("address y shape:", self._heteroData['address'].y.shape)
		#输出self._heteroData.y不是null的元素数
		print("address y elements:", self._heteroData['address'].y.numel())
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")



