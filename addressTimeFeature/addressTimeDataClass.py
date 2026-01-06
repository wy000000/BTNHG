from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeFeature.addressTimeFeatureClass import addressTimeFeatureClass
import pandas as pd
import os
import time
# import lz4.frame
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class addressTimeDataClass:
	def __init__(self,
		dataPath=BTNHGV2ParameterClass.dataPath):
		self._dataPath=dataPath

		self.addressTimeFeature_dataSet=self.get_address_time_feature_dataSet()
		# self.feature_dim=self.addressTimeFeature_dataSet.tensors[0].shape[-1]
		# self.seq_len = self.addressTimeFeature_dataSet.tensors[0].shape[-2]
		# self.num_classes = self.addressTimeFeature_dataSet.tensors[1].unique().numel()



		# self.addressTime_data_df=self._loadAddressTimeData()
		# self._zipMethod=lz4.frame
		# # 初始化 address_dict 为空字典
		# # 键：addressID
		# # 值：包含 clusterID 和 addressTimeFeatureCls 的字典
		# self.address_dict = self._processAddressTimeData()
		# self.addressTime_data_df=None

		# self.dataSet = self._build_address_time_feature_dataSet()
		# self.address_dict=None
		# self.save_address_time_feature_dataSet()

		# self.train_dataLoader, self.test_dataLoader\
		# 	=self.build_address_time_feature_trainLoader_testLoaser()
		# self.kFold_dataloaders=self.build_address_time_feature_KFold_dataLoaders()

		# self.KFoldIndices=self.build_address_time_feature_KFold_indices()
		
	def _loadAddressTimeData(self, dataPath:str=None)->pd.DataFrame:
		print("start read addressTimeData")
		if dataPath is None:
			dataPath=self._dataPath
		time1 = time.time()
		#读取所有数据
		addressTime_data_df = pd.read_csv(os.path.join(dataPath, "addressTimeData.csv"))
		time2=time.time()
		print("读取耗时：{}时{}分{}秒, addressTime_data_df.shape={}".format(int((time2-time1)//3600),
					int((time2-time1)//60), int((time2-time1)%60), addressTime_data_df.shape))
		return addressTime_data_df
	
	def _processAddressTimeData(self, addressTime_data_df):
		print("start process address time data")
		time1 = time.time()
		rowCount = addressTime_data_df.shape[0]
		
		# 使用字典存储addressTimeFeatureCls实例
		address_dict = {}

		# 使用itertuples()代替iterrows()，速度更快
		for i, row_tuple in enumerate(addressTime_data_df.itertuples(index=False)):
			addressID = row_tuple.addressID
			clusterID = row_tuple.clusterID
			
			if addressID not in address_dict:
				# 创建新实例并存储到字典
				addressTimeFeatureCls = addressTimeFeatureClass(row_tuple)
				addressTimeFeatureCls.process_address_time_features(row_tuple)
				address_dict[addressID] = {
					"clusterID": clusterID,
					"addressTimeFeatureCls": addressTimeFeatureCls
				}
			else:
				# 更新现有实例
				addressTimeFeatureCls = address_dict[addressID]["addressTimeFeatureCls"]
				addressTimeFeatureCls.process_address_time_features(row_tuple)
			if (i+1)%20001==0:
				print("已完成{}行，进度{:.2f}%".format(i, i/rowCount*100))
				# break
		print("更新差异特征")
		for addressID in address_dict:
			addressTimeFeatureCls = address_dict[addressID]["addressTimeFeatureCls"]
			addressTimeFeatureCls.update_diff_features()
		
		time2=time.time()
		print("时间特性处理完成。address_dict长度=",len(address_dict))
		# 输出处理时间 时：分：秒
		usedTime=time2-time1
		print("时间特性处理耗时：{}时{}分{}秒".format(int(usedTime//3600),
								int((usedTime)//60), int(usedTime%60)))
	
		# #输出self.address_dict前64个元素
		# print(list(self.address_dict.items())[:64])
		return address_dict
	
	def _build_address_time_feature_dataSet(self, addressDict):
		"""
		生成DataSet，包含地址的时间特征数据
		返回:
			TensorDataset: 包含时间特征数据的Dataset
		"""
		print("start build self.addressTimeFeature_dataSet")
		time1 = time.time()

		# # 收集所有 clusterID 和特征
		# cluster_ids = []
		# features_list = []
		# for v in addressDict.values():
		# 	cluster_ids.append(v["clusterID"])
		# 	features_list.append(v["addressTimeFeatureCls"].block_features.astype(np.float32))

		# # 转为 NumPy 数组
		# features_np = np.stack(features_list)   # shape: (N, d1, d2) 或 (N, d)
		# cluster_ids = np.array(cluster_ids)
		# time4=time.time()
		# print("创建features_np和cluster_ids耗时: {}时{}分{}秒"\
		# 	.format(int((time4-time1)//3600), int((time4-time1)//60), int((time4-time1)%60)))
		
		# time3=time.time()
		# # 将非连续 clusterID 映射为连续整数
		# unique_ids, labels_np = np.unique(cluster_ids, return_inverse=True)
		# time4=time.time()
		# print("标签连续化用时: {}时{}分{}秒"\
		# 	.format(int((time4-time3)//3600), int((time4-time3)//60), int((time4-time3)%60)))


		# # 转为 PyTorch Tensor
		# features = torch.from_numpy(features_np)
		# labels = torch.from_numpy(labels_np).long()

		# # 构建 TensorDataset
		# dataSet = TensorDataset(features, labels)

		#####################################################

		# print(time.strftime('%m-%d %H:%M:%S', time.localtime()))
		# 1. 获取地址总数和特征形状
		num_addresses = len(addressDict)
		if num_addresses == 0:
			return None
		
		# 获取第一个地址的特征形状
		first_address = next(iter(addressDict.values()))
		first_features = first_address["addressTimeFeatureCls"].block_features
		feature_shape = first_features.shape
		
		# 2. 预分配内存
		# 创建NumPy数组预分配特征和标签空间
		features_np = np.zeros((num_addresses, feature_shape[0],
						feature_shape[1]), dtype=np.float32)
		labels_np = np.zeros(num_addresses, dtype=np.int64)
		
		# 3. 填充数据
		address_items = list(addressDict.items())
		addressCount = len(address_items)
		for i, (addressID, value) in enumerate(address_items):
			addressTimeFeatureCls = value["addressTimeFeatureCls"]
			features_np[i] = addressTimeFeatureCls.block_features.astype(np.float32)
			labels_np[i] = value["clusterID"]
			if (i+1)%2001==0:
				print("已完成{}行，进度{:.2f}%".format(i, i/addressCount*100))
		
		# time3=time.time()
		# 将非连续 clusterID 映射为连续整数
		unique_ids, labels_np = np.unique(labels_np, return_inverse=True)
		# time4=time.time()
		# print("标签连续化用时: {}时{}分{}秒"\
		# 	.format(int((time4-time3)//3600), int((time4-time3)//60), int((time4-time3)%60)))
		
		# 4. 转换为Tensor
		features = torch.from_numpy(features_np)
		# features = features.unsqueeze(1)
		labels = torch.from_numpy(labels_np)
		
		# 5. 构造Dataset
		dataSet = TensorDataset(features, labels)
		time2=time.time()
		# print(time.strftime('%m-%d %H:%M:%S', time.localtime()))
		usedTime=time2-time1
		# 输出处理时间 时：分：秒
		print("build self.addressTimeFeature_dataSet used time: {}时{}分{}秒"\
			.format(int(usedTime//3600), int((usedTime)//60), int(usedTime%60)))
		self.addressTimeFeature_dataSet=dataSet
		return dataSet	

	def get_address_time_feature_dataSet(self, dataPath:str=None):
		"""
		获取地址时间特征数据集
		参数：
			dataPath: 数据集路径
		返回：
			TensorDataset: 地址时间特征数据集
		"""
		if dataPath is None:
			dataPath=self._dataPath

		dataDF=self._loadAddressTimeData(dataPath)
		addressDict=self._processAddressTimeData(dataDF)
		self.addressTimeFeature_dataSet=self._build_address_time_feature_dataSet(addressDict)
		return self.addressTimeFeature_dataSet

	# region
	# def _save_address_time_feature_dataSet(self, dataPath:str=None,
	# 		addressTimeFeature_dataSet_name=BTNHGV2ParameterClass.addressTimeFeature_dataSet_name):
	# 	if dataPath is None:
	# 		dataPath=self._dataPath
	# 	# 检查是否有数据集
	# 	if self.addressTimeFeature_dataSet is None:
	# 		print("没有数据集，请先构建数据集")
	# 		return None
		
	# 	time1 = time.time()
	# 	features, labels = self.addressTimeFeature_dataSet.tensors
	# 	# self.addressTimeFeature_dataSet = TensorDataset(features.to_sparse(), labels)

	# 	# 保存数据集		
	# 	print("start to save {}".format(addressTimeFeature_dataSet_name))
		
	# 	filePath=os.path.join(dataPath, addressTimeFeature_dataSet_name)
	# 	with self._zipMethod.open(filePath, "wb") as f:
	# 		torch.save(TensorDataset(features.to_sparse(), labels), f)
	# 	# self.addressTimeFeature_dataSet = TensorDataset(features.to_dense(), labels)
		
	# 	time2=time.time()
	# 	usedTime=time2-time1
		
	# 	# #输出保存文件的大小
	# 	# fileSize=os.path.getsize(filePath)
	# 	# print("{} file size: {}MB".format(addressTimeFeature_dataSet_name, fileSize/1024/1024))

	# 	# 保存文件用时 时：分：秒
	# 	print("save {} used time: {}时{}分{}秒".format(addressTimeFeature_dataSet_name,
	# 		int(usedTime//3600), int((usedTime)//60), int(usedTime%60)))
	
	# def load_address_time_feature_dataSet(self, dataPath:str=None,
	# 		addressTimeFeature_dataSet_name=BTNHGV2ParameterClass.addressTimeFeature_dataSet_name):

	# 	print("start load {}".format(addressTimeFeature_dataSet_name))
	# 	time1 = time.time()
	# 	if dataPath is None:
	# 		dataPath=self._dataPath
		
	# 	filePath=os.path.join(dataPath, addressTimeFeature_dataSet_name)
	# 	#文件不存在
	# 	if not os.path.exists(filePath):
	# 		print("not find {}, start to build it.".format(addressTimeFeature_dataSet_name))
	# 		dataDF=self._loadAddressTimeData(dataPath)
	# 		addressDict=self._processAddressTimeData(dataDF)
	# 		self.addressTimeFeature_dataSet=self._build_address_time_feature_dataSet(addressDict)
	# 		self._save_address_time_feature_dataSet(dataPath)
	# 	else:
	# 		with torch.serialization.safe_globals([torch.utils.data.dataset.TensorDataset]):
	# 			with self._zipMethod.open(filePath, "rb") as f:
	# 				self.addressTimeFeature_dataSet = torch.load(f)
	# 		features, labels = self.addressTimeFeature_dataSet.tensors
	# 		self.addressTimeFeature_dataSet = TensorDataset(features.to_dense(), labels)

	# 	time2=time.time()
	# 	usedTime=time2-time1
	# 	# 载入时间 时：分：秒
	# 	print("load {} used time: {}时{}分{}秒".format(addressTimeFeature_dataSet_name,
	# 		int(usedTime//3600), int((usedTime)//60), int(usedTime%60)))
		
	# 	return self.addressTimeFeature_dataSet
	# endregion

	def get_address_time_feature_trainLoader_testLoaser(self,
			train_size=BTNHGV2ParameterClass.train_size,
			batch_size=BTNHGV2ParameterClass.cnn_batch_size,
			shuffle=BTNHGV2ParameterClass.shuffle,
			resetSeed=BTNHGV2ParameterClass.resetSeed):
		print("start build trainLoader and testLoaser")
		time1 = time.time()
		# 检查是否有数据集
		if self.addressTimeFeature_dataSet is None:
			self.addressTimeFeature_dataSet = self.get_address_time_feature_dataSet()
		if self.addressTimeFeature_dataSet is None:
			return None, None
		
		# 直接获取tensor，避免不必要的转换
		features, labels = self.addressTimeFeature_dataSet.tensors

		# 随机种子
		randSeed = BTNHGV2ParameterClass.rand(resetSeed)

		# 将tensor转换为numpy以用于sklearn的train_test_split
		X = features.numpy() if not isinstance(features, np.ndarray) else features
		y = labels.numpy() if not isinstance(labels, np.ndarray) else labels
		X_train, X_test, y_train, y_test = train_test_split( X, y,
													train_size=train_size, # 测试集占比
													random_state=randSeed, # 保证可复现
													stratify=y # 保持类别比例一致
													)
		
		# 直接从numpy数组创建tensor，避免额外转换
		train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
		test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

		# 创建 DataLoader
		train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
		test_dataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		time2=time.time()
		usedTime=time2-time1
		# 输出处理时间 时：分：秒
		print("build trainLoader and testLoaser used time: {}时{}分{}秒"\
			.format(int(usedTime//3600), int((usedTime)//60), int(usedTime%60)))
		
		return train_dataLoader, test_dataLoader

	def get_address_time_feature_KFold_indices(self,
		k=BTNHGV2ParameterClass.kFold_k,
		batch_size=BTNHGV2ParameterClass.cnn_batch_size,
		shuffle=BTNHGV2ParameterClass.shuffle,
		resetSeed=BTNHGV2ParameterClass.resetSeed):
		"""
		生成用于K折交叉验证的DataLoader，包含训练和验证数据
		
		参数:
			k: 折数，默认为配置文件中的kFold_k
			shuffle: 是否打乱数据，默认为配置文件中的值
			random_state: 随机种子，确保结果可重复
			
		返回:
			list: 包含k个元组的列表，每个元组包含(train_dataLoader, val_dataLoader)
		"""
		print("start build KFold indices")
		time1 = time.time()
		# 检查是否有数据集
		if self.addressTimeFeature_dataSet is None:
			self.addressTimeFeature_dataSet = self._build_address_time_feature_dataSet()
		if self.addressTimeFeature_dataSet is None:
			return None
		
		# 1. 获取完整数据集
		dataset = self.addressTimeFeature_dataSet
		
		# 2. 获取特征和标签
		features, labels = self.addressTimeFeature_dataSet.tensors

		# 随机种子
		randSeed = BTNHGV2ParameterClass.rand(resetSeed)
		# 3. 初始化KFold
		skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=randSeed)
		# 预计算所有索引，避免重复切片操作
		KFoldIndices = list(skf.split(features, labels))

		# # 4. 生成每折的DataLoader
		# kfold_dataLoaders = []
		# for fold_idx, (train_idx, val_idx) in enumerate(KFoldIndices):
		# 	# 显示第k折
		# 	print(f"Processing fold {fold_idx + 1}/{k}")

			# # 创建训练和验证子集
			# train_dataset = TensorDataset(features[train_idx], labels[train_idx])
			# test_dataset = TensorDataset(features[val_idx], labels[val_idx])
			
			# # 创建DataLoader
			# train_dataLoader = DataLoader(train_dataset
			# 					,batch_size=batch_size
			# 					,shuffle=shuffle
			# 					# ,num_workers=0 # 根据需要可以增加此值
			# 					# ,pin_memory=False  # 如果有GPU可以设置为True
			# 					)
			
			# test_dataLoader = DataLoader(test_dataset
			# 					,batch_size=batch_size
			# 					,shuffle=False
			# 					# ,num_workers=0  # 根据需要可以增加此值
			# 					# ,pin_memory=False  # 如果有GPU可以设置为True
			# 					)
			# # 添加到结果列表
			# kfold_dataLoaders.append((train_dataLoader, test_dataLoader))
				
		time2=time.time()
		usedTime=time2-time1
		# 输出处理时间 时：分：秒
		print("build KFold dataLoaders used time: {}时{}分{}秒"\
			.format(int(usedTime//3600), int((usedTime)//60), int(usedTime%60)))
		
		return KFoldIndices


# atd=addressTimeDataClass()