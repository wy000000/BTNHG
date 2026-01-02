from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeFeature.addressTimeFeatureClass import addressTimeFeatureClass
import pandas as pd
import os
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold

class addressTimeDataClass:
	def __init__(self,
		dataPath=BTNHGV2ParameterClass.dataPath):
		self._dataPath=dataPath
		self.addressTime_data_df=self._loadAddressTimeData()

		# 初始化 address_dict 为空字典
		# 键：addressID
		# 值：包含 clusterID 和 addressTimeFeatureCls 的字典
		self.address_dict = {}
		self._processAddressTimeData()
		
	def _loadAddressTimeData(self, dataPath:str=None)->pd.DataFrame:
		print("start read addressTimeData")
		if dataPath is None:
			dataPath=self._dataPath
		#读取所有数据
		addressTime_data_df = pd.read_csv(os.path.join(dataPath, "addressTimeData-full.csv"))
		print("读取完成。addressTime_data_df.shape=",addressTime_data_df.shape)
		return addressTime_data_df
	
	def _processAddressTimeData(self):
		time1 = time.time()
		rowCount = self.addressTime_data_df.shape[0]
		
		# 使用字典存储addressTimeFeatureCls实例
		address_dict = self.address_dict

		# 使用itertuples()代替iterrows()，速度更快
		for i, row_tuple in enumerate(self.addressTime_data_df.itertuples(index=False)):
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
			if (i+1)%10001==0:
				print("已完成{}行，进度{:.2f}%".format(i, i/rowCount*100))
				# break
		print("更新差异特征")
		for addressID in address_dict:
			addressTimeFeatureCls = address_dict[addressID]["addressTimeFeatureCls"]
			addressTimeFeatureCls.update_diff_features()
		
		time2=time.time()
		print("时间特性处理完成。address_dict长度=",len(address_dict))
		# 输出处理时间 时：分：秒
		print("时间特性处理耗时：{}时{}分{}秒".format(int((time2-time1)//3600), int((time2-time1)//60), int((time2-time1)%60)))
		#输出当前时间
		print("当前时间：{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
	
		# #输出self.address_dict前64个元素
		# print(list(self.address_dict.items())[:64])
		return address_dict
	
	def get_address_time_feature_dataSet(self):
		"""
		生成用于训练的DataLoader，包含地址的时间特征数据
		返回:
			TensorDataset: 包含时间特征数据的Dataset
		"""
		
		# 1. 获取地址总数和特征形状
		num_addresses = len(self.address_dict)
		if num_addresses == 0:
			return None
		
		# 获取第一个地址的特征形状
		first_address = next(iter(self.address_dict.values()))
		first_features = first_address["addressTimeFeatureCls"].block_features
		feature_shape = first_features.shape
		
		# 2. 预分配内存
		# 创建NumPy数组预分配特征和标签空间
		features_np = np.zeros((num_addresses, feature_shape[0],
						feature_shape[1]), dtype=np.float32)
		labels_np = np.zeros(num_addresses, dtype=np.int64)
		
		# 3. 填充数据
		for i, (addressID, value) in enumerate(self.address_dict.items()):
			addressTimeFeatureCls = value["addressTimeFeatureCls"]
			features_np[i] = addressTimeFeatureCls.block_features.astype(np.float32)
			labels_np[i] = value["clusterID"]
		
		# 4. 转换为Tensor
		features = torch.tensor(features_np, dtype=torch.float32)
		labels = torch.tensor(labels_np, dtype=torch.long)
		
		# 5. 构造Dataset和DataLoader
		dataset = TensorDataset(features, labels)
		# dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)		
		return dataset
	
	def get_address_time_feature_dataLoader(self,
			batch_size=BTNHGV2ParameterClass.batch_size,
			shuffle=BTNHGV2ParameterClass.shuffle):
		"""
		生成用于训练的DataLoader，包含地址的时间特征数据
		返回:
			DataLoader: 包含时间特征数据的DataLoader
		"""
		# 1. 获取完整数据集
		dataset = self.get_address_time_feature_dataSet()
		if dataset is None:
			return None
		
		# 2. 构造DataLoader
		dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)		
		return dataLoader

	def get_address_time_feature_KFold_dataLoader(self,
		k=BTNHGV2ParameterClass.kFold_k,
		batch_size=BTNHGV2ParameterClass.batch_size,
		shuffle=BTNHGV2ParameterClass.shuffle,
		random_state=BTNHGV2ParameterClass.randSeed):
		"""
		生成用于K折交叉验证的DataLoader，包含训练和验证数据
		
		参数:
			k: 折数，默认为配置文件中的kFold_k
			batch_size: 批量大小，默认为配置文件中的值
			shuffle: 是否打乱数据，默认为配置文件中的值
			random_state: 随机种子，确保结果可重复
			
		返回:
			list: 包含k个元组的列表，每个元组包含(train_dataLoader, val_dataLoader)
		"""
		# 1. 获取完整数据集
		dataset = self.get_address_time_feature_dataSet()
		if dataset is None:
			return None
		
		# 2. 获取特征和标签
		features, labels = dataset.tensors
		
		# 3. 初始化KFold
		skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
		
		# 4. 生成每折的DataLoader
		kfold_dataLoaders = []
		for train_idx, val_idx in skf.split(features, labels):
			# 创建训练和验证子集
			train_dataset = TensorDataset(features[train_idx], labels[train_idx])
			val_dataset = TensorDataset(features[val_idx], labels[val_idx])
			
			# 创建DataLoader
			train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
			val_dataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
			
			# 添加到结果列表
			kfold_dataLoaders.append((train_dataLoader, val_dataLoader))
		
		return kfold_dataLoaders	


# atd=addressTimeDataClass()