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
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
# import sys
time2 = time.time()
# print("import used time: ", time2 - time1)
# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

class BTNHGV2LoaderClass:
	def __init__(self, heteroData=None):
		"""
		初始化 BTNHGV2LoaderClass 类
		Args:
			heteroData: 异构图数据对象
		"""
		if heteroData is None:
			return None
		self._heteroData=heteroData
		# self._train_size=train_size
		# self._batch_size=batch_size
		# self._shuffle=shuffle
		# self._isResetSeed=isResetSeed
	
	def getTrainLoaderAndTestLoader(self, train_size=BTNHGV2ParameterClass.train_size,
								batch_size=BTNHGV2ParameterClass.batch_size,
								shuffle=BTNHGV2ParameterClass.shuffle, isResetSeed=False):
		"""
		划分数据集为训练集和测试集,并返回DataLoader类型的train_loader和test_loader
		Args:
			train_size: 训练集比例，默认值为 BTNHGV2ParameterClass.train_size
			batch_size: 批次大小，默认值为 BTNHGV2ParameterClass.batch_size
			shuffle: 是否打乱数据集，默认值为 BTNHGV2ParameterClass.shuffle
			isResetSeed: 是否重置随机种子，默认值为 False
		Returns:
			train_loader: 训练数据加载器(DataLoader类型)
			test_loader: 测试数据加载器(DataLoader类型)
		"""
		# print("flag")
		print("start split dataset to train and test")
		time1 = time.time()
		labeled_address_indices = torch.where(self._heteroData['address'].y != -1)[0]
		self._heteroData.num_labeled = len(labeled_address_indices)

		if self._heteroData.num_labeled == 0:
			return None, None

		labels = self._heteroData['address'].y[labeled_address_indices].numpy()

		randSeed = BTNHGV2ParameterClass.rand(isResetSeed)
		print(f"randSeed:{randSeed}, type:{type(randSeed)}")
		# if(isResetSeed):
		# 	random.seed(randSeed)
		# 	np.random.seed(randSeed)
		# 	torch.manual_seed(randSeed)
		# 	if torch.cuda.is_available():
		# 		torch.cuda.manual_seed_all(randSeed)

		train_indices, test_indices = train_test_split(
			np.arange(self._heteroData.num_labeled),
			train_size=train_size,
			stratify=labels,
			random_state=randSeed
		)
		# if(isResetSeed):
		# 	train_indices = np.sort(train_indices)
		# 	test_indices = np.sort(test_indices)
		
		# train_indices, test_indices = _stratified_train_test_split(
		# 	np.arange(self._heteroData.num_labeled), labels
		# 	1-paramsClass.train_size,
		# 	stratify=True,
		# 	shuffle=paramsClass.shuffle,
		# 	random_state=randSeed
		# )

		# 转换为真实节点索引
		train_nodes = labeled_address_indices[train_indices]
		test_nodes = labeled_address_indices[test_indices]

		# 构造数据集 (假设特征是 x，标签是 y)
		x = self._heteroData['address'].x
		y = self._heteroData['address'].y

		train_dataset = TensorDataset(x[train_nodes], y[train_nodes])
		test_dataset = TensorDataset(x[test_nodes], y[test_nodes])

		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		time2 = time.time()
		print("划分数据集信息")
		print(f"训练集大小: {len(train_indices)} ({len(train_indices)/self._heteroData.num_labeled:.2%})")
		print(f"测试集大小: {len(test_indices)} ({len(test_indices)/self._heteroData.num_labeled:.2%})")
		print(f"划分数据集用时: {time2 - time1}")
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

		return train_loader, test_loader
	
###################################################################
	# def getTrainTestMask(self, train_size=BTNHGV2ParameterClass.train_size,
	# 						batch_size=BTNHGV2ParameterClass.batch_size,
	# 						shuffle=BTNHGV2ParameterClass.shuffle,
	# 						isResetSeed=False):
			
	# 		# print("start split dataset to train and test")
	# 		time1 = time.time()
	# 		labeled_address_indices = torch.where(self._heteroData['address'].y != -1)[0]
	# 		num_labeled = len(labeled_address_indices)

	# 		if num_labeled == 0:
	# 			return None, None

	# 		labels = self._heteroData['address'].y[labeled_address_indices].numpy()

	# 		randSeed = BTNHGV2ParameterClass.rand(isResetSeed)

	# 		train_indices, test_indices = train_test_split(
	# 			np.arange(num_labeled),
	# 			train_size=train_size,
	# 			stratify=labels,
	# 			random_state=randSeed)

	# 		train_mask = torch.zeros(self._heteroData['address'].num_nodes, dtype=torch.bool)
	# 		test_mask = torch.zeros(self._heteroData['address'].num_nodes, dtype=torch.bool)

	# 		train_mask[labeled_address_indices[train_indices]] = True
	# 		test_mask[labeled_address_indices[test_indices]] = True			

	# 		time2 = time.time()
	# 		# 3. 打印划分信息
	# 		print("划分数据集信息")
	# 		print(f"训练集大小: {len(train_indices)} ({len(train_indices)/num_labeled:.2%})")
	# 		print(f"测试集大小: {len(test_indices)} ({len(test_indices)/num_labeled:.2%})")
	# 		print(f"划分数据集用时: {time2 - time1}")
	# 		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
	# 		return train_mask, test_mask
	
	###################################################################
	# def _stratified_train_test_split(X, y, test_size=0.25,
	# 								stratify=True, shuffle=True, random_state=None):
	# 	"""
	# 	自定义 train_test_split，支持 stratify 和 shuffle 同时控制。
		
	# 	参数：
	# 	----------
	# 	X : array-like, shape (n_samples, n_features)
	# 		特征矩阵
	# 	y : array-like, shape (n_samples,)
	# 		标签
	# 	test_size : float
	# 		测试集比例
	# 	stratify : bool
	# 		是否分层抽样
	# 	shuffle : bool
	# 		是否打乱数据
	# 	random_state : int 或 None
	# 		随机种子
		
	# 	返回：
	# 	----------
	# 	X_train, X_test, y_train, y_test
	# 	"""		
	# 	rng = check_random_state(random_state)
	# 	X = np.array(X)
	# 	y = np.array(y)
	# 	n_samples = len(y)
	# 	n_test = int(np.floor(test_size * n_samples))
		
	# 	if stratify:
	# 		# 按类别分层
	# 		train_idx, test_idx = [], []
	# 		classes, y_indices = np.unique(y, return_inverse=True)
	# 		for cls in range(len(classes)):
	# 			cls_idx = np.where(y_indices == cls)[0]
	# 			n_cls_test = int(np.floor(test_size * len(cls_idx)))
	# 			cls_test_idx = rng.choice(cls_idx, size=n_cls_test, replace=False)
	# 			cls_train_idx = np.setdiff1d(cls_idx, cls_test_idx)
	# 			train_idx.extend(cls_train_idx)
	# 			test_idx.extend(cls_test_idx)
	# 	else:
	# 		# 不分层，直接随机划分
	# 		indices = np.arange(n_samples)
	# 		test_idx = rng.choice(indices, size=n_test, replace=False)
	# 		train_idx = np.setdiff1d(indices, test_idx)
		
	# 	# 是否打乱
	# 	if shuffle:
	# 		rng.shuffle(train_idx)
	# 		rng.shuffle(test_idx)
	# 	# else:
	# 	# 	# 保持原始顺序
	# 	# 	train_idx = np.sort(train_idx)
	# 	# 	test_idx = np.sort(test_idx)
		
	# 	# return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
	# 	return train_idx, test_idx
	# ##################################################
		


# ##################################################
# d=BTNHGDatasetClass()
# d._loadBTNHGV2Data()
# td11, td12 = d.getTrainLoaderAndTestLoader()
# td21, td22 = d.getTrainLoaderAndTestLoader()
# #比较td11.dataset和td21.dataset是否相同

# # 导入必要的库

# # 方法1：检查对象标识（内存地址）
# print("对象标识是否相同（内存地址）:")
# print("td11.dataset is td21.dataset:", td11.dataset is td21.dataset)  # 应该是False，因为是不同实例

# # 方法2：检查数据集大小
# print("\n数据集大小是否相同:")
# print("数据集长度比较:", len(td11.dataset) == len(td21.dataset))

# # 方法3：比较数据集的张量内容
# print("\n数据集张量内容是否相同:")
# if hasattr(td11.dataset, 'tensors') and hasattr(td21.dataset, 'tensors'):
#     # 修复：使用len(td11.dataset.tensors)获取张量数量
#     print(f"td11.dataset包含{len(td11.dataset.tensors)}个张量")
#     print(f"td21.dataset包含{len(td21.dataset.tensors)}个张量")
    
#     # 比较每个张量是否相等
#     all_tensors_equal = True
#     for i, (t1, t2) in enumerate(zip(td11.dataset.tensors, td21.dataset.tensors)):
#         is_equal = torch.equal(t1, t2)
#         print(f"张量{i+1}相等: {is_equal}")
#         if not is_equal:
#             all_tensors_equal = False
#             break
    
#     print("所有张量内容相同:", all_tensors_equal)
# else:
#     print("无法访问tensors属性")

# # 方法4：检查前几个样本是否相同（如果数据集不为空）
# print("\n前3个样本是否相同:")
# if len(td11.dataset) > 0 and len(td21.dataset) > 0:
#     sample_count = min(3, len(td11.dataset), len(td21.dataset))
#     for i in range(sample_count):
#         sample1 = td11.dataset[i]
#         sample2 = td21.dataset[i]
        
#         if isinstance(sample1, tuple) and isinstance(sample2, tuple):
#             # 如果样本是元组(x,y)，分别比较
#             x_equal = torch.equal(sample1[0], sample2[0])
#             y_equal = torch.equal(sample1[1], sample2[1])
#             print(f"样本{i+1}特征相等: {x_equal}, 标签相等: {y_equal}")
#         else:
#             # 如果样本不是元组，直接比较
#             equal = torch.equal(sample1, sample2)
#             print(f"样本{i+1}相等: {equal}")
# else:
#     print("数据集为空，无法比较样本")

# # 方法5：检查训练数据索引是否相同（基于之前的分析）
# # 从文件内容看，数据集是通过索引创建的，所以可以检查索引是否一致
# print("\n数据集内容总结:")
# print(f"td11.dataset类型: {type(td11.dataset).__name__}")
# print(f"td21.dataset类型: {type(td21.dataset).__name__}")
# print(f"数据集大小相同: {len(td11.dataset) == len(td21.dataset)}")







############################################

	# def get_data_loaders(self, batch_size=paramsClass.batch_size, randSeed=None):
	# 	"""返回 train_loader 和 test_loader"""
	# 	train_mask, test_mask = self._split_dataset(randSeed=randSeed)

	# 	if train_mask is None or test_mask is None:
	# 		return None, None

	# 	# 根据 mask 获取索引
	# 	train_indices = train_mask.nonzero(as_tuple=True)[0]
	# 	test_indices = test_mask.nonzero(as_tuple=True)[0]

	# 	# 构造子数据集
	# 	train_dataset = self._heteroData['address'].index_select(train_indices)
	# 	test_dataset = self._heteroData['address'].index_select(test_indices)

	# 	# 构造 DataLoader
	# 	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=paramsClass.shuffle)
	# 	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	# 	# train_loader.train_mask = train_mask
	# 	# test_loader.test_mask = test_mask
		
	# 	return train_loader, test_loader

	# def get_heteroData(self):
	# 	"""获取训练集DataLoader"""
	# 	# 1. 加载数据
	# 	self._loadBTNHGV2Data()
	# 	# 2. 划分数据集
	# 	# self._split_dataset()
	# 	# 3. 返回划分好的数据集
	# 	print("返回划分好的数据集")
	# 	print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
	# 	return self._heteroData
	# #生成trainLoader, testLoader

	#生成一个函数返回trainLoader, testLoader
	

#测试分布
class TestHeteroDataClass:
	def test_heteroData(self, heteroData):
		# 创建数据集实例
		# dataset = BTNHGDatasetClass()
		# heteroData=dataset.get_heteroData()

		# 验证标签分布是否相同
		print("\n=== 训练集和测试集标签分布验证 ===")

		# 计算训练集和测试集标签
		mask_train, mask_test = self._split_dataset()
		# mask_train = heteroData['address'].train_mask
		# mask_test = heteroData['address'].test_mask

		
		# 过滤掉无效标签(-1)并计算有效标签的掩码
		valid_train_mask = mask_train & (heteroData['address'].y != -1)
		valid_test_mask = mask_test & (heteroData['address'].y != -1)
		
		# 获取有效标签
		valid_train_labels = heteroData['address'].y[valid_train_mask]
		valid_test_labels = heteroData['address'].y[valid_test_mask]
		
		# 计算训练集标签分布
		if valid_train_labels.numel() > 0:
			train_label_counts = torch.bincount(valid_train_labels)
			train_label_dist = train_label_counts.float() / train_label_counts.sum()
			
			# 获取有正样本的标签索引
			positive_indices = torch.nonzero(train_label_counts, as_tuple=True)[0]
			positive_counts = train_label_counts[positive_indices]
			positive_dists = train_label_dist[positive_indices]
		
		# 计算测试集标签分布
		if valid_test_labels.numel() > 0:
			test_label_counts = torch.bincount(valid_test_labels)
			test_label_dist = test_label_counts.float() / test_label_counts.sum()
			
			# 获取有正样本的标签索引
			test_positive_indices = torch.nonzero(test_label_counts, as_tuple=True)[0]
			test_positive_counts = test_label_counts[test_positive_indices]
			test_positive_dists = test_label_dist[test_positive_indices]
		
		# 打印统计信息
		print(f"训练集大小: {len(valid_train_labels)} ({len(valid_train_labels)/heteroData.num_labeled:.2%})")
		print(f"测试集大小: {len(valid_test_labels)} ({len(valid_test_labels)/heteroData.num_labeled:.2%})")
		
		# 打印训练集标签分布
		print("\n训练集标签分布:")
		if valid_train_labels.numel() > 0:
			# 使用向量化操作生成格式化字符串
			train_lines = [f"标签 {idx}: {count} 个 ({dist:.2%})" 
						for idx, count, dist in zip(positive_indices.tolist(), 
													positive_counts.tolist(), 
													positive_dists.tolist())]
			print('\n'.join(train_lines))
		
		# 打印测试集标签分布
		print("\n测试集标签分布:")
		if valid_test_labels.numel() > 0:
			# 使用向量化操作生成格式化字符串
			test_lines = [f"标签 {idx}: {count} 个 ({dist:.2%})" 
						for idx, count, dist in zip(test_positive_indices.tolist(), 
													test_positive_counts.tolist(), 
													test_positive_dists.tolist())]
			print('\n'.join(test_lines))
		#print now time
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")