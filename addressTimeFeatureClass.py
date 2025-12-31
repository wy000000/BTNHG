import pandas as pd
import numpy as np
from BTNHGV2ParameterClass import BTNHGV2ParameterClass

class addressTimeFeatureClass:
	def __init__(self,
				# address_time_feat_df:pd.DataFrame,
				# i:int):
				row:pd.Series):
		self.minBlockID=272375
		self.maxBlockID=277995
		self.spanOfBlocks=self.maxBlockID-self.minBlockID+1
		self.addressBlockFeatureCount=13

		self.row=row
		self.addressID=self.row["addressID"]
		self.clusterID=self.row["clusterID"]
		# self.blockMap={}
		# self.blockFeatures = np.zeros((self.spanOfBlocks, self.addressBlockFeatureCount))
		blockFeatureDict = { "blockID": -1, "coinCount": 0, "coinAmount": 0, "inCoinCount": 0,\
							"inCoinAmount": 0, "outCoinCount": 0, "outCoinAmount": 0, "diffCoinCount": 0,\
							"diffCoinAmount": 0, "txCount": 0, "inTxCount": 0, "outTxCount": 0,\
							"diffTxCount": 0 }
		# 创建 DataFrame
		self.blockFeatureDF = pd.DataFrame([blockFeatureDict]*self.spanOfBlocks)
		
	
	def _process_address_time_features(self):
		row=self.row
		


			
	


	