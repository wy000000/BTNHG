from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeFeatureClass import addressTimeFeatureClass
import pandas as pd
import os
import time

class addressTimeDataClass:
	# addressCount=546649
	def __init__(self,
		dataPath=BTNHGV2ParameterClass.dataPath):
		self._dataPath=dataPath
		self.addressTime_data_df=self._loadAddressTimeData()

		# 初始化 address_dict 为空字典
		# 键：addressID
		# 值：包含 clusterID 和 addressTimeFeatureCls 的字典
		self.address_dict = {}
		self.processAddressTimeData()
		#将address_dict中的addressTimeFeatureCls.block_features展平追加到address_dict,
		#保留block_features各项的名称
		
		
		
		# self.address_time_feature_dict = {}
	
	def _loadAddressTimeData(self, dataPath:str=None)->pd.DataFrame:
		print("start read addressTimeData")
		if dataPath is None:
			dataPath=self._dataPath
		#读取所有数据
		addressTime_data_df = pd.read_csv(os.path.join(dataPath, "addressTimeData.csv"))
		print("读取完成。addressTime_data_df.shape=",addressTime_data_df.shape)
		return addressTime_data_df
	
	# 使用itertuples()代替iterrows()，速度更快
	def processAddressTimeData(self):
		time1 = time.time()
		rowCount = self.addressTime_data_df.shape[0]
		
		# 使用字典存储addressTimeFeatureCls实例
		address_dict = self.address_dict		
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
	

	
	


atd=addressTimeDataClass()