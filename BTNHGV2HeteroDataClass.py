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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import check_random_state
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from sklearn.utils.class_weight import compute_class_weight
import csv
# import sys
time2 = time.time()
# print("import used time: ", time2 - time1)
# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

class BTNHGV2HeteroDataClass():
	"""
	比特币交易网络数据集类，用于处理异构图数据的划分和加载
	"""
	def __init__(self, heteroData=None,
			dataPath=BTNHGV2ParameterClass.dataPath):
		"""
		初始化 BTNHGV2HeteroDataClass 类
		Args:
			heteroData: 异构图数据对象 (HeteroData)，如果为 None，则从文件加载
			dataPath: 数据文件路径，默认值为 BTNHGV2ParameterClass.dataPath			
		"""
		self.dataPath=dataPath
		self.heteroData=None
		self._address_id_map = None
		self._coin_id_map = None
		self._tx_id_map = None
		self._cluster_id_map = None
		self.class_weight=None #在getTrainTestMask()中计算
		self.cluster_count=None		

		if heteroData is not None:
			self.heteroData=heteroData
		else:
			self._loadBTNHGV2Data(dataPath=self.dataPath)
								# debugMode=self.debugMode)
			self.getTrainTestMask()
			self.getTrainTestMaskKFold()

	def _loadBTNHGV2Data(self,
						dataPath=None):
		"""
		从文件加载比特币交易网络数据集,放入self.heteroData
		Args:
			dataPath: 数据文件路径，默认值为 BTNHGV2ParameterClass.dataPath
		"""
		print("start construct HeteroData")
		time1 = time.time()
		if dataPath is None:
			dataPath=self.dataPath
		
		# 1. 读取数据
		print("start read data")
		time1 = time.time()
		
		#读取所有数据
		addr_feat_df = pd.read_csv(os.path.join(dataPath, "addressFeature.csv"))
		coin_feat_df = pd.read_csv(os.path.join(dataPath, "coinFeature.csv"))
		tx_feat_df   = pd.read_csv(os.path.join(dataPath, "TxFeature.csv"))
		edge_df      = pd.read_csv(os.path.join(dataPath, "hgEdgeV2.csv"))
		################################
		time2 = time.time()
		print(f"读取数据用时: {time2 - time1}")
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

		# 2. 建立 ID 映射
		print("construct ID map and 构建节点特征矩阵")
		time1 = time.time()
		address_ids = addr_feat_df['addressID'].unique()
		coin_ids    = coin_feat_df['coinID'].unique()
		tx_ids      = tx_feat_df['txID'].unique()

		self._address_id_map = {id_: i for i, id_ in enumerate(address_ids)}
		self._coin_id_map    = {id_: i for i, id_ in enumerate(coin_ids)}
		self._tx_id_map      = {id_: i for i, id_ in enumerate(tx_ids)}

		# 3. 初始化 HeteroData
		self.heteroData = HeteroData()

		# 4. 构建节点特征矩阵
		self.heteroData['address'].x = torch.tensor(
			addr_feat_df.drop(columns=['addressID']).fillna(-1).values, dtype=torch.float
		)
		self.heteroData['coin'].x = torch.tensor(
			coin_feat_df.drop(columns=['coinID']).fillna(0).values, dtype=torch.float
		)
		self.heteroData['tx'].x = torch.tensor(
			tx_feat_df.drop(columns=['txID']).fillna(0).values, dtype=torch.float
		)
		time2 = time.time()
		# print(f"构建ID map and 节点特征矩阵用时: {time2 - time1}")
		# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

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
		self.heteroData['address', 'addr_to_coin', 'coin'].edge_index \
			= build_edge(edge_df, 'addressID', 'coinID', self._address_id_map, self._coin_id_map)
		
		self.heteroData['tx', 'tx_to_coin', 'coin'].edge_index \
			= build_edge(edge_df, 'txID_coin', 'coinID', self._tx_id_map, self._coin_id_map)

		self.heteroData['coin', 'coin_to_tx', 'tx'].edge_index \
			= build_edge(edge_df, 'coinID', 'coin_txID', self._coin_id_map, self._tx_id_map)
		time2 = time.time()
		print(f"构建边关系用时: {time2 - time1}")
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
		#################################################################
		# 6. 给 address 节点加标签
		print("给 address 节点加标签")
		time1 = time.time()
		# 构建聚类ID映射
		unique_cluster_ids = edge_df["clusterID"].dropna().unique()
		self._cluster_id_map = {cid: idx for idx, cid in enumerate(unique_cluster_ids)}

		address_y = torch.full((len(self._address_id_map),), -1, dtype=torch.long)
		# 使用向量化操作
		valid_cluster = edge_df[['addressID', 'clusterID']].dropna()
		# 过滤出有效的地址和聚类ID
		valid_mask = valid_cluster['addressID'].isin(self._address_id_map.keys())
		valid_data = valid_cluster[valid_mask]
		# 批量转换和赋值
		if not valid_data.empty:
			# 根据 addressID 找到对应的索引位置
			indices = [self._address_id_map[addr] for addr in valid_data['addressID']]
			# 将 clusterID 转换为 self._cluster_id_map 中的索引值
			clusters = [self._cluster_id_map[c] for c in valid_data['clusterID']]
			# 赋值给 address_y
			address_y[indices] = torch.tensor(clusters, dtype=torch.long)
		self.heteroData['address'].y = address_y
		time2 = time.time()
		# print("给 address 节点加标签用时: ", time2 - time1)
		
		# #转成 无向图
		# print("转成无向图")
		time1 = time.time()
		# print(f"当前边数: {self._getEdgeCount()}")

		#无向与自环二选一。无向信息更全。有向丢信息，效果不好。
		self._to_undirected()
		# self._add_self_loops_for_isolated_nodes()

		# 确保所有 edge_index 都是连续的
		for store in self.heteroData.edge_stores:
			store.edge_index = store.edge_index.contiguous()
		# print(f"当前边数: {self._getEdgeCount()}")
		time2 = time.time()
		# print(f"转无向图用时: {time2 - time1}")
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

		self._countCluster()
		self.printClusterCount()

	def _add_self_loops_for_isolated_nodes(self):
	# 遍历所有节点类型
		for ntype in self.heteroData.node_types:
			num_nodes = self.heteroData[ntype].num_nodes

			# 检查该节点类型是否有入边
			has_in_edge = False
			for (src, rel, dst), edge_index in self.heteroData.edge_index_dict.items():
				if dst == ntype and edge_index.size(1) > 0:
					has_in_edge = True
					break

			# 如果没有入边，则添加自环
			if not has_in_edge:
				self_loop_index = torch.arange(num_nodes)
				edge_index = torch.stack([self_loop_index, self_loop_index], dim=0)
				self.heteroData[(ntype, 'self', ntype)].edge_index = edge_index		

	def _to_undirected(self):
		"""
		将self.heteroData有向异构图转为无向图:为每条边添加反向边
		"""
		# print("make undirected graph")
		time1 = time.time()
		# 注意：遍历时要用 list() 包裹，否则在迭代过程中修改字典会出问题
		for (src, rel, dst) in list(self.heteroData.edge_types):
			edge_index = self.heteroData[(src, rel, dst)].edge_index

			# 生成反向边索引
			edge_index_rev = edge_index.flip(0)

			# 定义反向关系名
			rel_rev = rel + "_rev"

			# 如果反向边不存在，则添加
			if (dst, rel_rev, src) not in self.heteroData.edge_types:
				self.heteroData[(dst, rel_rev, src)].edge_index = edge_index_rev

				# 如果原边有特征，也复制
				if "edge_attr" in self.heteroData[(src, rel, dst)]:
					self.heteroData[(dst, rel_rev, src)].edge_attr = \
						self.heteroData[(src, rel, dst)].edge_attr.clone()
		time2 = time.time()
		#输出self.heteroData的所有边的数量
		# print(f"make undirected graph用时: {time2 - time1}")
		# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

	def _getEdgeCount(self):
		"""
		获取self.heteroData的所有边的数量
		返回:int, self.heteroData的所有边的数量
		"""
		total_edges = 0
		for edge_type in self.heteroData.edge_types:
			num_edges = self.heteroData[edge_type].edge_index.size(1)
			# print(f"  {edge_type}: {num_edges}")
			total_edges += num_edges
		return total_edges
	
	def getTrainTestMask(self, train_size=BTNHGV2ParameterClass.train_size,
							shuffle=BTNHGV2ParameterClass.shuffle,
							resetSeed=BTNHGV2ParameterClass.resetSeed):	
		#regoin
		"""
		为异构图数据中的address节点生成训练集和测试集掩码。
		同时生成种类权重。
		
		该函数从异构图数据中筛选出带有有效标签（非-1）的address节点，
		使用分层抽样方法将数据按照指定比例划分为训练集和测试集，
		确保训练集和测试集中各类别的分布与原始数据保持一致。
		
		参数
		----------
		train_size : float, 可选
			训练集所占比例，默认值从BTNHGV2ParameterClass获取
		shuffle : bool, 可选
			是否打乱数据顺序，默认值从BTNHGV2ParameterClass获取
		resetSeed : bool, 可选
			是否重置随机种子，控制是否每次调用生成相同的划分结果，
			默认值从BTNHGV2ParameterClass获取
		
		返回
		----------
		tuple of torch.Tensor or (None, None)
			返回两个布尔类型的PyTorch张量：
			- train_mask: 长度为address节点总数的布尔张量，True表示该节点属于训练集
			- test_mask: 长度为address节点总数的布尔张量，True表示该节点属于测试集
			如果没有找到带标签的节点，则返回(None, None)    
		"""
		#endregion

		time1 = time.time()

		# 获取所有有标签的节点索引
		labeled_address_indices = torch.where(self.heteroData['address'].y != -1)[0]
		num_labeled = len(labeled_address_indices)

		if num_labeled == 0:
			return None, None

		# 获取标签
		labels = self.heteroData['address'].y[labeled_address_indices].cpu().numpy()

		# 随机种子
		randSeed = BTNHGV2ParameterClass.rand(resetSeed)

		# 划分训练集和测试集
		train_indices, test_indices = train_test_split(
												np.arange(num_labeled),
												train_size=train_size,
												stratify=labels,
												random_state=randSeed)

		# 创建 mask
		train_mask = torch.zeros(self.heteroData['address'].num_nodes, dtype=torch.bool)
		test_mask = torch.zeros(self.heteroData['address'].num_nodes, dtype=torch.bool)

		train_mask[labeled_address_indices[train_indices]] = True
		test_mask[labeled_address_indices[test_indices]] = True

		self.heteroData['address'].train_mask = train_mask
		self.heteroData['address'].test_mask = test_mask

		# ✅ 计算 class_weight（基于训练集标签）
		train_labels = self.heteroData['address'].y[train_mask].cpu().numpy()
		classes = np.unique(train_labels)
		weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
		self.class_weight = torch.tensor(weights, dtype=torch.float)

		time2 = time.time()
		#打印划分信息
		# print(f"训练集大小: {len(train_indices)} ({len(train_indices)/num_labeled:.2%})")
		# print(f"测试集大小: {len(test_indices)} ({len(test_indices)/num_labeled:.2%})")
		# print(f"划分数据集用时: {time2 - time1}")
		# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
		print("划分数据集完成")
		return train_mask, test_mask
	
	def getTrainTestMaskKFold(self,
							kFold_k=BTNHGV2ParameterClass.kFold_k,
							shuffle=BTNHGV2ParameterClass.shuffle,
							resetSeed=BTNHGV2ParameterClass.resetSeed):
		# 获取所有有标签的节点索引
		labeled_address_indices = torch.where(self.heteroData['address'].y != -1)[0]
		num_labeled = len(labeled_address_indices)

		if num_labeled == 0:
			return []

		# 获取标签
		labels = self.heteroData['address'].y[labeled_address_indices].cpu().numpy()

		# 随机种子
		randSeed = BTNHGV2ParameterClass.rand(resetSeed)

		# 定义分层K折
		skf = StratifiedKFold(n_splits=kFold_k, shuffle=shuffle, random_state=randSeed)

		# 保存每折的mask
		masks = []

		for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(num_labeled), labels)):
			# 创建 mask
			train_mask = torch.zeros(self.heteroData['address'].num_nodes, dtype=torch.bool)
			test_mask = torch.zeros(self.heteroData['address'].num_nodes, dtype=torch.bool)

			train_mask[labeled_address_indices[train_idx]] = True
			test_mask[labeled_address_indices[test_idx]] = True

			masks.append((train_mask, test_mask))
		self.heteroData['address'].kFold_masks = masks
		print(f"kFold划分完成")
		return masks

	def _countCluster(self):
		"""
		统计每个类别的数量
		"""
		# 获取所有标签
		y_all = self.heteroData['address'].y
		# 过滤掉无效标签（假设 -1 表示无效）
		valid_y = y_all[y_all != -1].cpu().numpy()
		# 统计每个类别的数量，并存储到对象（字典）中
		classes, counts = np.unique(valid_y, return_counts=True)
		# 拼接成二维数组：第一行是类别，第二行是数量
		self.class_count = np.vstack((classes, counts))

	def printClusterCount(self):
		"""
		打印每个类别的数量
		"""
		# 指定宽度，例如 6
		width = 6
		# 第一行输出类别，每个元素占用固定宽度
		print("类别: ", end="")
		for c in self.class_count[0]:
			print(f"{c:<{width}}", end="")
		print()  # 换行
		# 第二行输出数量，每个元素占用固定宽度
		print("数量: ", end="")
		for cnt in self.class_count[1]:
			print(f"{cnt:<{width}}", end="")
		print()		
	
	def get_clusterID(self, addressID):
		"""		
		根据 addressID 查找对应的 clusterID。
		
		参数:
			addressID: 原始地址ID (如 addressID=22389567, clusterID=6438509)
		
		返回:
			clusterID (int)，如果没有标签则返回 None
		"""
		if addressID not in self._address_id_map:
			return(f"addressID: {addressID} 不在映射表中")		
		idx = self._address_id_map[addressID]                # 找到节点索引
		clusterIDIndex = int(self.heteroData['address'].y[idx])  # 查找对应标签
		#反查 clusterID
		clusterID = [clusterID for clusterID, idx in self._cluster_id_map.items()\
					if idx == clusterIDIndex]		
		return clusterID
	
	def get_addressID(self, i):
		#打印self._address_id_map的大小
		print(f"self._address_id_map的大小: {len(self._address_id_map)}")
		print(f"self._address_id_map的第{i}个元素: {self._address_id_map[i]}")
		return self._address_id_map[i]
