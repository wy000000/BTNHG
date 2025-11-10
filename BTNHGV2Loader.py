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
	random_state=42

class BTNHGDataset(Dataset):
	"""
	比特币交易网络数据集类，用于处理异构图数据的划分和加载
	"""
	def __init__(self, train_size=paramsClass.train_size,
				 random_state=paramsClass.random_state,
				 dataPath=paramsClass.dataPath):
		super().__init__()
		self.data=self._loadBTNHGV2Data(dataPath)
		self.train_size = train_size
		self.random_state = random_state        
		# 初始化时就划分数据集，避免重复计算
		self.train_data, self.test_data = self._split_dataset()

	def _loadBTNHGV2Data(dataPath):
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

		# 构建边
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
		data['address'].y = address_y

		time2 = time.time()
		print("给 address 节点加标签时间: ", time2 - time1)
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
		#输出data的所有类型的结点及其特征矩阵的形状
		print("address node types:", data.node_types)
		print("address:"+str(data['address'].x.shape))
		print("coin:"+str(data['coin'].x.shape))
		print("tx:"+str(data['tx'].x.shape))
		# 输出data的所有类型的边及其边索引的形状
		print("edge types:", data.edge_types)
		print("address-coin:"+str(data['address', 'addr_to_coin', 'coin'].edge_index.shape))
		print("tx-coin:"+str(data['tx', 'tx_to_coin', 'coin'].edge_index.shape))
		print("coin-tx:"+str(data['coin', 'coin_to_tx', 'tx'].edge_index.shape))
		#输出data.y的形状
		print("address y shape:", data['address'].y.shape)
		#输出data.y不是null的元素数
		print("address y elements:", data['address'].y.numel())

		return data
		
	def _split_dataset(self):
		"""内部方法：划分数据集"""
		# 获取有标签的address节点索引
		labeled_address_indices = torch.where(self.data['address'].y != -1)[0]
		num_labeled = len(labeled_address_indices)
		
		if num_labeled == 0:
			return self.data.clone(), self.data.clone()
			
		# 获取有标签的address节点的标签
		labels = self.data['address'].y[labeled_address_indices].numpy()
		
		# 分层采样划分训练集和测试集
		train_indices, test_indices = train_test_split(
			np.arange(num_labeled),
			train_size=self.train_size,
			stratify=labels,
			random_state=self.random_state
		)
		
		# 创建掩码和数据分割（代码与原实现相同）
		train_mask = torch.zeros(self.data['address'].num_nodes, dtype=torch.bool)
		test_mask = torch.zeros(self.data['address'].num_nodes, dtype=torch.bool)
		
		# 设置掩码
		train_mask[self.labeled_address_indices[train_indices]] = True
		test_mask[self.labeled_address_indices[test_indices]] = True
		
		# 克隆数据并应用掩码
		train_data = self.data.clone()
		test_data = self.data.clone()
		
		train_data['address'].train_mask = train_mask
		train_data['address'].test_mask = torch.zeros_like(train_mask)
		
		test_data['address'].train_mask = torch.zeros_like(test_mask)
		test_data['address'].test_mask = test_mask
		
		return train_data, test_data
	
	def get_train_loader(self, batch_size=8, shuffle=True):
		"""获取训练集DataLoader"""
		return DataLoader(
			[self.train_data],
			batch_size=batch_size,
			shuffle=shuffle,
			follow_batch=['address', 'tx', 'coin']
		)
	
	def get_test_loader(self, batch_size=8):
		"""获取测试集DataLoader"""
		return DataLoader(
			[self.test_data],
			batch_size=batch_size,
			shuffle=False,
			follow_batch=['address', 'tx', 'coin']
		)
	
	# 保留原有的len()和get()方法
	def len(self):
		# 整个图作为一个数据实例
		return 1
	
	def get(self, idx):
		return self.data

# 创建数据集实例
dataset = BTNHGDataset(data)

# 划分训练集和测试集
train_data, test_data = dataset.split_dataset(train_size=0.8)

# 创建DataLoader
train_loader = DataLoader(
	[train_data],  # PyG的DataLoader期望数据列表
	batch_size=8,
	shuffle=True,
	follow_batch=['address', 'tx', 'coin']  # 指定需要跟踪批次的节点类型
)

test_loader = DataLoader(
	[test_data],
	batch_size=8,
	shuffle=False,  # 测试集通常不打乱
	follow_batch=['address', 'tx', 'coin']
)

# 对于大图的替代方案：使用NeighborLoader进行邻居采样
def create_neighbor_loaders(train_data, test_data, batch_size=8, shuffle=True, num_neighbors=[10, 5]):
	"""
	使用NeighborLoader创建训练集和测试集的数据加载器（适用于大图）
	"""
	# 获取训练集中有标签的address节点索引
	train_nodes = torch.where(train_data['address'].train_mask)[0]
	test_nodes = torch.where(test_data['address'].test_mask)[0]
	
	# 创建NeighborLoader
	train_loader = NeighborLoader(
		train_data,
		num_neighbors=num_neighbors,  # 每层采样的邻居数量
		batch_size=batch_size,
		input_nodes=('address', train_nodes),  # 从address节点开始采样
		shuffle=shuffle,
	)
	
	test_loader = NeighborLoader(
		test_data,
		num_neighbors=num_neighbors,
		batch_size=batch_size,
		input_nodes=('address', test_nodes),
		shuffle=False,
	)
	
	return train_loader, test_loader

# 如果图很大，可以使用下面的代码替换上面的DataLoader创建：
# train_loader, test_loader = create_neighbor_loaders(train_data, test_data, batch_size=8, shuffle=True)

# 验证标签分布是否相同
print("\n=== 训练集和测试集标签分布验证 ===")
train_labels = train_data['address'].y[train_data['address'].train_mask]
test_labels = test_data['address'].y[test_data['address'].test_mask]

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

