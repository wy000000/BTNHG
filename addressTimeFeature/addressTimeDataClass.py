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

		# print(time.strftime('%m-%d %H:%M:%S', time.localtime()))
		# 1. 获取地址总数和特征形状
		num_addresses = len(addressDict)
		if num_addresses == 0:
			raise ValueError("addressDict is empty")
		
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

	def compress_address_time_feature_dataSet(self):
		"""
		压缩地址时间特征数据集：
		对每个样本，过滤掉除第0列外全为0的时间步（时间步级过滤）
		
		返回：
		TensorDataset：填充后的地址时间特征数据集
		"""
		dataSet = self.addressTimeFeature_dataSet
		if dataSet is None:
			raise ValueError("addressTimeFeature_dataSet is None")
		
		features, labels = dataSet.tensors
		num_samples = features.shape[0]
		print(f"压缩前：num_samples={num_samples},\n"
			+f"features.shape={features.shape}, labels.shape={labels.shape}\n开始压缩")

		time1 = time.time()

		# 统计压缩前的总时间步（特征行数）
		total_before = features.shape[0] * features.shape[1]
		
		# 直接进行时间步级过滤（跳过样本级过滤）

		filtered_samples = []
		# total_after = 0  # 统计压缩后的总特征行数
		
		for i in range(num_samples):
			sample_features = features[i]
			has_non_zero_step = torch.any(sample_features[:, 1:] != 0, dim=1)
			filtered_sample = sample_features[has_non_zero_step]
			
			# 实现：在filtered_sample最左边添加一列全1
			# 创建全1列，形状为[时间步数量, 1]
			ones_column = torch.ones(filtered_sample.shape[0], 1, device=filtered_sample.device)
			# 在维度1（特征维度）上拼接全1列和原特征
			filtered_sample = torch.cat([ones_column, filtered_sample], dim=1)
			
			#for i>0，如果第i行的第1个元素值减第i-1行的第1个元素值不等于1，则设置第i行的第0个元素值为差值
			if filtered_sample.shape[0] > 1:  # 至少有两行数据才需要处理
				# 计算相邻行第1个元素（索引1）的差值
				diffs = filtered_sample[1:, 1] - filtered_sample[:-1, 1]
				# 创建掩码：差值不等于1的位置
				mask = diffs != 1
				# 将满足条件的行的第0个元素（索引0）设置为差值
				filtered_sample[1:, 0][mask] = diffs[mask]

			filtered_samples.append(filtered_sample)
			# total_after += filtered_sample.shape[0]  # 累加每个样本保留的时间步数
			if (i+1)%2001==0:
				print("已完成{}行，进度{:.2f}%".format(i, i/num_samples*100))
		
		# 找出所有样本中的最大时间步长度
		max_len = max(sample.shape[0] for sample in filtered_samples)		
		
		# 获取特征维度
		feature_dim = filtered_samples[0].shape[1]
		
		# 创建填充后的特征张量
		padded_features = torch.zeros((num_samples, max_len, feature_dim), device=features.device)
		
		# 对每个样本进行填充
		for i, sample in enumerate(filtered_samples):
			padded_features[i, :sample.shape[0], :] = sample
		
		# 创建TensorDataset
		padded_dataset = TensorDataset(padded_features, labels)

		time2 = time.time()
		usedTime=time2-time1

		print(f"压缩后：padded_features.shape={padded_features.shape}, labels.shape={labels.shape}")

		#计算填充后的压缩率
		total_after = padded_features.shape[0] * padded_features.shape[1]
		print(f"压缩减少：{1-(total_after/total_before):.2%}")

		# 输出处理时间 时：分：秒
		print("compress_address_time_feature_dataSet used time: {}时{}分{}秒"\
			.format(int(usedTime//3600), int((usedTime)//60), int(usedTime%60)))
		
		self.addressTimeFeature_dataSet = padded_dataset		
		return padded_dataset

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
		# cnn_batch_size=BTNHGV2ParameterClass.cnn_batch_size,
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
			# 					,batch_size=cnn_batch_size
			# 					,shuffle=shuffle
			# 					# ,num_workers=0 # 根据需要可以增加此值
			# 					# ,pin_memory=False  # 如果有GPU可以设置为True
			# 					)
			
			# test_dataLoader = DataLoader(test_dataset
			# 					,batch_size=cnn_batch_size
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