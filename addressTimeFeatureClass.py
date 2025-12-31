import pandas as pd
import numpy as np
from BTNHGV2ParameterClass import BTNHGV2ParameterClass

class addressTimeFeatureClass:
	minBlockID=272375
	maxBlockID=277995
	spanOfBlocks=maxBlockID-minBlockID+1
	addressBlockFeatureCount=13

	def __init__(self
				, address_time_feat_df:pd.DataFrame
				, rowNo:int):		
		self.addressID=address_time_feat_df.loc[rowNo, "addressID"]
		self.clusterID=address_time_feat_df.loc[rowNo, "clusterID"]		
		blockFeatureDict = { "blockID": -1, "coinCount": 0, "coinAmount": 0, "inCoinCount": 0,\
							"inCoinAmount": 0, "outCoinCount": 0, "outCoinAmount": 0, "diffCoinCount": 0,\
							"diffCoinAmount": 0, "txCount": 0, "inTxCount": 0, "outTxCount": 0,\
							"diffTxCount": 0 }
		# 创建 DataFrame
		self.blockFeatureDF = pd.DataFrame([blockFeatureDict]*self.spanOfBlocks)		
	
	def process_address_time_features(self
								, address_time_feat_df:pd.DataFrame
								, rowNo:int):
		self.process_inTx(address_time_feat_df, rowNo)
		self.process_outTx(address_time_feat_df, rowNo)

	def process_inTx(self
					, address_time_feat_df:pd.DataFrame
					, rowNo:int):
		minBlockID=addressTimeFeatureClass.minBlockID
		#address_time_feat_df是原始数据，处理当前行，rowNo为当前行号
		inBlockID=address_time_feat_df.iloc[rowNo, "inBlockID"]
		# inBlockID is nan
		if pd.isna(inBlockID):
			return False
		value=address_time_feat_df.loc[rowNo, "value"]
		#更新address的inBlockID的block数据
		self.blockFeatureDF.loc[inBlockID-minBlockID, "blockID"]=inBlockID
		self.blockFeatureDF.loc[inBlockID-minBlockID, "coinCount"]+=1
		self.blockFeatureDF.loc[inBlockID-minBlockID, "coinAmount"]+=value
		self.blockFeatureDF.loc[inBlockID-minBlockID, "inCoinCount"]+=1
		self.blockFeatureDF.loc[inBlockID-minBlockID, "inCoinAmount"]+=value
		self.blockFeatureDF.loc[inBlockID-minBlockID, "txCount"]+=1
		self.blockFeatureDF.loc[inBlockID-minBlockID, "inTxCount"]+=1
		return True
	
	def process_outTx(self
					, address_time_feat_df:pd.DataFrame
					, rowNo:int):
		minBlockID=addressTimeFeatureClass.minBlockID
		#address_time_feat_df是原始数据，处理当前行，rowNo为当前行号		
		outBlockID=address_time_feat_df.loc[rowNo, "outBlockID"]
		# outBlockID is nan
		if pd.isna(outBlockID):
			return False
		value=address_time_feat_df.loc[rowNo, "value"]
		#更新address的outBlockID的block数据
		self.blockFeatureDF.loc[outBlockID-minBlockID, "blockID"]=outBlockID
		self.blockFeatureDF.loc[outBlockID-minBlockID, "coinCount"]+=1
		self.blockFeatureDF.loc[outBlockID-minBlockID, "coinAmount"]+=value
		self.blockFeatureDF.loc[outBlockID-minBlockID, "outCoinCount"]+=1
		self.blockFeatureDF.loc[outBlockID-minBlockID, "outCoinAmount"]+=value
		self.blockFeatureDF.loc[outBlockID-minBlockID, "txCount"]+=1
		self.blockFeatureDF.loc[outBlockID-minBlockID, "outTxCount"]+=1
		return True