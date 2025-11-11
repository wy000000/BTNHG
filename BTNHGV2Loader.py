print("start import")
import time
time1 = time.time()
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
import numpy as np
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
time2 = time.time()
print("import time: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

#定义一个参数类，使用静态变量来保存程序中用到的各种参数
class paramsClass():
	dataPath=r"D:\BTNHG\BTNHGV2"
	train_size=0.8
	产生一个随机数
	random_sate=np.random.randint(0, int.maxsize)
	# random_state=42
	shuffle=True

class BTNHGDatasetClass(Dataset):
	"""
	比特币交易网络数据集类，用于处理异构图数据的划分和加载
	"""
	def __init__(self):
		super().__init__()
		self.data=None

	def _loadBTNHGV2Data(self):
		print("start construct HeteroData")
		time1 = time.time()

		# 1. 读取数据
		print("start read data")
		time1 = time.time()
		addr_feat_df = pd.read_csv(os.path.join(paramsClass.dataPath, "addressFeature.csv"))
		coin_feat_df = pd.read_csv(os.path.join(paramsClass.dataPath, "coinFeature.csv"))
		tx_feat_df   = pd.read_csv(os.path.join(paramsClass.dataPath, "TxFeature.csv"))
		edge_df      = pd.read_csv(os.path.join(paramsClass.dataPath, "hgEdgeV2.csv"))
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
		self.data = HeteroData()

		# 4. 构建节点特征矩阵（直接用 replace + values）
		self.data['address'].x = torch.tensor(
			addr_feat_df.drop(columns=['addressID']).values, dtype=torch.float
		)
		self.data['coin'].x = torch.tensor(
			coin_feat_df.drop(columns=['coinID']).values, dtype=torch.float
		)
		self.data['tx'].x = torch.tensor(
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

		# 构建边
		self.data['address', 'addr_to_coin', 'coin'].edge_index = build_edge(edge_df, 'addressID', 'coinID',
																		address_id_map, coin_id_map)

		self.data['tx', 'tx_to_coin', 'coin'].edge_index = build_edge(edge_df, 'txID_coin', 'coinID',
																tx_id_map, coin_id_map)

		self.data['coin', 'coin_to_tx', 'tx'].edge_index = build_edge(edge_df, 'coinID', 'coin_txID',
																coin_id_map, tx_id_map)
		time2 = time.time()
		print(f"构建边关系时间: {time2 - time1}")
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
		self.data['address'].y = address_y

		time2 = time.time()
		print("给 address 节点加标签时间: ", time2 - time1)
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
		#输出self.data的所有类型的结点及其特征矩阵的形状
		print("address node types:", self.data.node_types)
		print("address:"+str(self.data['address'].x.shape))
		print("coin:"+str(self.data['coin'].x.shape))
		print("tx:"+str(self.data['tx'].x.shape))
		# 输出self.data的所有类型的边及其边索引的形状
		print("edge types:", self.data.edge_types)
		print("address-coin:"+str(self.data['address', 'addr_to_coin', 'coin'].edge_index.shape))
		print("tx-coin:"+str(self.data['tx', 'tx_to_coin', 'coin'].edge_index.shape))
		print("coin-tx:"+str(self.data['coin', 'coin_to_tx', 'tx'].edge_index.shape))
		#输出self.data.y的形状
		print("address y shape:", self.data['address'].y.shape)
		#输出self.data.y不是null的元素数
		print("address y elements:", self.data['address'].y.numel())
		
	def _split_dataset(self):
		"""划分数据集为训练集和测试集"""
		print("start split dataset to train and test")
		time1 = time.time()
		labeled_address_indices = torch.where(self.data['address'].y != -1)[0]
		num_labeled = len(labeled_address_indices)

		if num_labeled == 0:
			return None, None

		labels = self.data['address'].y[labeled_address_indices].numpy()

		train_indices, test_indices = train_test_split(
			np.arange(num_labeled),
			train_size=self.train_size,
			stratify=labels,
			random_state=self.random_state)

		train_mask = torch.zeros(self.data['address'].num_nodes, dtype=torch.bool)
		test_mask = torch.zeros(self.data['address'].num_nodes, dtype=torch.bool)

		train_mask[labeled_address_indices[train_indices]] = True
		test_mask[labeled_address_indices[test_indices]] = True

		# 直接在原始 data 上添加掩码
		self.data['address'].train_mask = train_mask
		self.data['address'].test_mask = test_mask		
		time2 = time.time()
		# 3. 打印划分信息
		print("划分数据集信息")
		print(f"训练集大小: {len(train_indices)} ({len(train_indices)/num_labeled:.2%})")
		print(f"测试集大小: {len(test_indices)} ({len(test_indices)/num_labeled:.2%})")
		print(f"划分数据集时间: {time2 - time1}")
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
	
	def get_dataLoader(self):
		"""获取训练集DataLoader"""
		# 1. 加载数据
		self._loadBTNHGV2Data()
		# 2. 划分数据集
		self._split_dataset()
		# 3. 返回划分好的数据集
		print("返回划分好的数据集")
		return self.data

# 创建数据集实例
dataset = BTNHGDatasetClass()
dataLoader=dataset.get_dataLoader()

# 验证标签分布是否相同
print("\n=== 训练集和测试集标签分布验证 ===")
train_labels = dataLoader['address'].y[dataLoader['address'].train_mask]
test_labels = dataLoader['address'].y[dataLoader['address'].test_mask]

# 计算训练集标签分布
train_label_counts = torch.bincount(train_labels[train_labels != -1])
train_label_dist = train_label_counts.float() / train_label_counts.sum()

# 计算测试集标签分布
test_label_counts = torch.bincount(test_labels[test_labels != -1])
test_label_dist = test_label_counts.float() / test_label_counts.sum()

print(f"训练集大小: {len(train_labels[train_labels != -1])} ({len(train_labels[train_labels != -1])/dataset.num_labeled:.2%})")
print(f"测试集大小: {len(test_labels[test_labels != -1])} ({len(test_labels[test_labels != -1])/dataset.num_labeled:.2%})")
print("\n训练集标签分布:")
for i, count in enumerate(train_label_counts):
	if count > 0:
		print(f"标签 {i}: {count} 个 ({train_label_dist[i]:.2%})")

print("\n测试集标签分布:")
for i, count in enumerate(test_label_counts):
	if count > 0:
		print(f"标签 {i}: {count} 个 ({test_label_dist[i]:.2%})")

# 保存划分好的数据（可选）
# torch.save(train_data, 'train_data.pt')
# torch.save(test_data, 'test_data.pt')

