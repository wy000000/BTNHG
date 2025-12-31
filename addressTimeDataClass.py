from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeFeatureClass import addressTimeFeatureClass
import pandas as pd
import os

class addressTimeDataClass:
	def __init__(self,
		dataPath=BTNHGV2ParameterClass.dataPath):
		self._dataPath=dataPath
		self.addressTime_feat_df=self._loadAddressTimeFeature()
		#定义一个addressTimeFeatureClass类型的list
		self.addressTimeFeatureCls_list=[]
	
	def _loadAddressTimeFeature(self, dataPath:str=None)->pd.DataFrame:
		print("start read addressTimeData")
		if dataPath is None:
			dataPath=self._dataPath
		#读取所有数据
		addressTime_feat_df = pd.read_csv(os.path.join(dataPath, "addressTimeData.csv"))
		return addressTime_feat_df
	
	def processAddressTimeFeature(self):
		#取self.address_time_feat_df的长度
		length=self.address_time_feat_df.shape[0]
		i=0
		while i<length:
			addressTimeFeatureCls=addressTimeFeatureClass(self.addressTime_feat_df, i)
			i=addressTimeFeatureCls.getAddressTimeFeature()
			

			

	
atd=addressTimeDataClass()
			

	


		


