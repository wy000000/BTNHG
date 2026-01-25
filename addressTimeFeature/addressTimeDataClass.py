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
		dataPath=BTNHGV2ParameterClass.dataPath,
		compress:bool=BTNHGV2ParameterClass.compress_dataSet,
		compress_padding:bool=BTNHGV2ParameterClass.compress_padding,
		noisy_0:bool=BTNHGV2ParameterClass.noisy_0,
		log_addressTimeFeature_amount:bool=BTNHGV2ParameterClass.log_addressTimeFeature_amount,
		try_read_save_addressTimeFeature_dataSet:bool\
			=BTNHGV2ParameterClass.try_read_save_addressTimeFeature_dataSet,
		):
	
		self._dataPath:str=dataPath
		# 是否尝试读取保存addressTimeFeature_dataSet
		# if true, 尝试读取保存的addressTimeFeature_dataSet, 如果不存在, 则重新构建
		# if false, 不尝试读取保存的addressTimeFeature_dataSet, 直接重新构建
		self._try_read_save_addressTimeFeature_dataSet:bool\
			=try_read_save_addressTimeFeature_dataSet
		# 是否采用压缩数据
		self._compress:bool=compress
		# 是否对压缩数据进行padding
		self._compress_padding:bool=compress_padding
		# 是否padding 0 添加轻微噪音扰动
		self._noisy_0:bool=noisy_0
		# 是否对addressTimeFeature中的amount类数值进行log变换，避免值过大值
		self._log_addressTimeFeature_amount:bool=log_addressTimeFeature_amount

		self.addressTimeFeature_dataSet:TensorDataset=self.get_address_time_feature_dataSet()

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
	
	def get_address_time_feature_dataSet(self)->TensorDataset:
		"""
		获取地址时间特征数据集
		参数：
			dataPath: 数据集路径
		返回：
			TensorDataset: 地址时间特征数据集
		"""
		dataPath=self._dataPath

		loadSuccessfully=False
		if self._try_read_save_addressTimeFeature_dataSet:
			loadSuccessfully=self.load_tensor_dataset()

		if not loadSuccessfully: # 加载失败，重新构建
			print(f"load addressTimeFeature_dataSet failed, try to build it")

			dataDF=self._loadAddressTimeData(dataPath)
			addressDict=self._processAddressTimeData(dataDF)
			self.addressTimeFeature_dataSet=self._build_address_time_feature_dataSet(addressDict)
			if self._compress:
				self.addressTimeFeature_dataSet=self.compress_address_time_feature_dataSet()

			if self._try_read_save_addressTimeFeature_dataSet:
				self.save_tensor_dataset()

		return self.addressTimeFeature_dataSet
	
	def _processAddressTimeData(self, addressTime_data_df:pd.DataFrame)->dict:		
		"""
		处理地址时间数据，将每个地址的时间特征聚合到一个字典中
		
		此方法遍历地址时间数据，为每个地址创建或更新 addressTimeFeatureClass 实例，
		处理时间特征，并最终更新差异特征。
		
		参数:
		addressTime_data_df (pd.DataFrame): 包含地址时间数据的DataFrame，
		至少包含 addressID 和 clusterID 列
		
		返回:
		dict: 地址字典，键为 addressID，值为包含以下内容的字典：
		- clusterID (int): 地址所属的聚类ID
		- addressTimeFeatureCls (addressTimeFeatureClass): 包含处理后时间特征的实例
		
		执行流程:
		1. 遍历输入的DataFrame中的每一行
		2. 对每个addressID，创建或更新addressTimeFeatureClass实例
		3. 调用process_address_time_features方法处理时间特征
		4. 定期打印处理进度
		5. 遍历所有地址，更新差异特征
		6. 打印处理结果和耗时统计
		"""
		
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
				addressTimeFeatureCls = addressTimeFeatureClass(row=row_tuple,
						log_addressTimeFeature_amount=self._log_addressTimeFeature_amount)
				addressTimeFeatureCls.process_address_time_features(row=row_tuple)
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
	
	def _build_address_time_feature_dataSet(self, addressDict:dict)->TensorDataset:
		"""
		将地址字典中的时间特征数据构建成 PyTorch 的 TensorDataset
		
		此方法从 addressDict 中提取所有地址的时间特征和聚类标签，
		构建成用于深度学习模型训练的标准化数据集。
		
		参数:
			addressDict (dict): 地址字典，键为 addressID，值为包含以下内容的字典：
				- clusterID (int): 地址所属的聚类ID
				- addressTimeFeatureCls (addressTimeFeatureClass): 包含处理后时间特征的实例
		
		返回:
			TensorDataset: PyTorch 数据集，包含两个张量：
				- 第一个张量：时间特征数据，形状为 [num_addresses, feature_shape[0], feature_shape[1]]
				- 第二个张量：连续化处理后的聚类标签，形状为 [num_addresses]
		
		执行流程:
			1. 检查地址字典是否为空，为空则抛出 ValueError
			2. 获取地址总数和第一个地址的特征形状
			3. 预分配 NumPy 数组空间存储特征和标签
			4. 遍历地址字典，填充特征和标签数据
			5. 将非连续的 clusterID 映射为连续整数
			6. 将 NumPy 数组转换为 PyTorch Tensor
			7. 构造并返回 TensorDataset
		
		注意事项:
			- 此方法会将非连续的 clusterID 转换为连续整数，以便模型训练
			- 特征数据类型为 float，标签数据类型为 int64
			- 方法会在内部设置 self.addressTimeFeature_dataSet 属性
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
		
		# 将非连续 clusterID 映射为连续整数
		unique_ids, labels_np = np.unique(labels_np, return_inverse=True)
		
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

	def compress_address_time_feature_dataSet(self,
				minBlockID: int = addressTimeFeatureClass.minBlockID)->TensorDataset:
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
		
		filtered_samples = []
		# total_after = 0  # 统计压缩后的总特征行数
		
		for i in range(num_samples):
			# 第i个样本的特征
			sample_features = features[i]
			#去掉sample_features中第0列等于-1的行
			filtered_sample = sample_features[sample_features[:, 0] != -1]

			# 在filtered_sample最左边添加一列全1
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
			
			#将样本的第0个时间步的第0个元素值设为 filtered_sample[0, 1]-minBlockID
			filtered_sample[0, 0] = filtered_sample[0, 1] - minBlockID
			
			if self._compress_padding: #填充样本################
				filtered_sample = self.pad_compress_address_time_feature_dataSet(filtered_sample)

			filtered_samples.append(filtered_sample)

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

	def pad_compress_address_time_feature_dataSet(self
						, feature:torch.Tensor
						, kernel_size=BTNHGV2ParameterClass.cnn_kernel_height						
						)->torch.Tensor:
		"""
		对地址时间特征数据集进行填充处理
		
		参数:
			feature: 一个sample的时间特征张量，形状为[seq_len, num_features]
			kernel_size: 卷积核大小，用于确定填充长度的阈值
		返回：
			processed_feature: 填充后的时间特征张量
		
		处理规则:
			- 对feature的每个时间步的第0列（时间步标识）值d进行处理
			- 如果d < 1，抛出ValueError异常
			- 如果1 < d <= kernel_size，padding_size = d - 1
			- 否则，padding_size = kernel_size
			- 在当前时间步前面添加padding_size个时间步
			- 每个填充的时间步第0个值为d，其他特征值为0
		"""
		if feature is None:
			raise ValueError("feature cannot be None")
		
		seq_len, num_features = feature.shape
		processed_segments = []
		
		for i in range(seq_len):
			# 获取当前时间步的第0列值d
			d = feature[i, 0]
			
			# 检查d是否小于0
			if d < 0:
				raise ValueError(f"时间步标识d必须大于等于0，但当前值第{i}行为{d}")
			
			# 计算padding_size
			if d > 1:
				if d <= kernel_size:
					padding_size = int(d - 1)
				else:
					padding_size = kernel_size
				
				# 创建填充时间步
				# 形状为[padding_size, num_features]
				padding = torch.zeros((padding_size, num_features), device=feature.device)

				# 每个填充时间步的第0个值为d
				padding[:, 0] = d

				# 为除了第0列之外的其他列添加轻微噪音扰动
				if self._noisy_0:
					# 生成轻微噪音，使用标准差为0.01的正态分布
					noise = torch.randn((padding_size, num_features - 1), device=feature.device) * 0.00001
					# 将噪音添加到除了第0列之外的其他列
					padding[:, 1:] = noise
				
				# 将填充时间步添加到当前时间步前面
				processed_segments.append(padding)
			
			# 添加当前时间步
			processed_segments.append(feature[i].unsqueeze(0))
		
		# 拼接所有处理后的时间步
		processed_feature = torch.cat(processed_segments, dim=0)
		
		return processed_feature
	
	def dataSet_log1p(self)->TensorDataset:
		features, labels = self.addressTimeFeature_dataSet.tensors
		# 对addressTimeFeature_dataSet的每个样本的第1列及之后的所有列进行log1p变换
		# features[:, :, 1:] = torch.log1p(features[:, :, 1:])
		# 对addressTimeFeature_dataSet的每个样本的所有列进行log1p变换
		features = torch.log1p(features)

		self.addressTimeFeature_dataSet = TensorDataset(features, labels)
		return TensorDataset(features, labels)




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

	def save_tensor_dataset(self):
		"""
		将 TensorDataset 保存到文件。
		保存内容包括 dataset.tensors 中的所有张量。
		"""
		dataset = self.addressTimeFeature_dataSet
		if dataset is None:
			raise ValueError("addressTimeFeature_dataSet is None, please build it first.")

		fullPath = os.path.join(self._dataPath, BTNHGV2ParameterClass.addressTimeFeature_dataSet_name)
		
		print(f"saving addressTimeFeature_dataSet to {fullPath}")

		data_dict = {f"tensor_{i}": t for i, t in enumerate(dataset.tensors)}

		torch.save(data_dict, fullPath)

		print ("saved")
		return fullPath

	def load_tensor_dataset(self, path: str=None) -> bool:
		"""
		从文件中读取 TensorDataset。
		加载成功返回 True，失败返回 False。
		"""
		try:
			if path is None:
				path = self._dataPath
			fullPath = os.path.join(self._dataPath, BTNHGV2ParameterClass.addressTimeFeature_dataSet_name)
		
			print(f"loading addressTimeFeature_dataSet from {fullPath}")
		
			data_dict = torch.load(fullPath)
	
			tensors = [data_dict[key] for key in sorted(data_dict.keys())]
	
			self.addressTimeFeature_dataSet = TensorDataset(*tensors)
		
			print ("loaded")
			return True
		
		# except FileNotFoundError:
		# 	print(f"错误：数据集文件不存在 - {fullPath}")
		# 	return False
		except Exception as e:
			print(f"加载 tensorDataset 失败: {e}")
			return False




# atd=addressTimeDataClass()