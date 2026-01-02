import pandas as pd
import numpy as np
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from typing import Any

class addressTimeFeatureClass:
	#region
	minBlockID = 272375
	maxBlockID = 277995
	spanOfBlocks = maxBlockID - minBlockID + 1
	addressBlockFeatureCount = 13
	
	# 定义特征名称和对应的索引
	FEATURE_NAMES = [
		"blockID", "coinCount", "coinAmount", "inCoinCount", "inCoinAmount",
		"outCoinCount", "outCoinAmount", "diffCoinCount", "diffCoinAmount",
		"txCount", "inTxCount", "outTxCount", "diffTxCount"]
	
	# 预计算特征索引，提高访问速度
	BLOCK_ID_IDX = FEATURE_NAMES.index("blockID")
	COIN_COUNT_IDX = FEATURE_NAMES.index("coinCount")
	COIN_AMOUNT_IDX = FEATURE_NAMES.index("coinAmount")
	IN_COIN_COUNT_IDX = FEATURE_NAMES.index("inCoinCount")
	IN_COIN_AMOUNT_IDX = FEATURE_NAMES.index("inCoinAmount")
	OUT_COIN_COUNT_IDX = FEATURE_NAMES.index("outCoinCount")
	OUT_COIN_AMOUNT_IDX = FEATURE_NAMES.index("outCoinAmount")
	DIF_COIN_COUNT_IDX = FEATURE_NAMES.index("diffCoinCount")
	DIF_COIN_AMOUNT_IDX = FEATURE_NAMES.index("diffCoinAmount")	
	TX_COUNT_IDX = FEATURE_NAMES.index("txCount")
	IN_TX_COUNT_IDX = FEATURE_NAMES.index("inTxCount")
	OUT_TX_COUNT_IDX = FEATURE_NAMES.index("outTxCount")
	DIF_TX_COUNT_IDX = FEATURE_NAMES.index("diffTxCount")

	#endregion

	def __init__(self, row: Any):
		self.addressID = row.addressID
		self.clusterID = row.clusterID
		
		# 使用 NumPy 数组替代 DataFrame，初始化所有值为0
		# 形状：(spanOfBlocks, addressBlockFeatureCount)
		self.block_features = np.zeros((self.spanOfBlocks, self.addressBlockFeatureCount),
							dtype=np.float32)
		
		# 初始化 blockID 为 -1
		self.block_features[:, self.BLOCK_ID_IDX] = -1

	def process_address_time_features(self, row: Any):
		self._process_inTx(row)
		self._process_outTx(row)
		return

	def _process_inTx(self, row: Any):
		inBlockID = row.inBlockID
		if pd.isna(inBlockID):
			return False
			
		value = row.value
		idx = int(inBlockID - self.minBlockID)
		
		# 更新特征数组（比 DataFrame.loc 快得多）
		self.block_features[idx, self.BLOCK_ID_IDX] = inBlockID
		self.block_features[idx, self.COIN_COUNT_IDX] += 1
		self.block_features[idx, self.COIN_AMOUNT_IDX] += value
		self.block_features[idx, self.IN_COIN_COUNT_IDX] += 1
		self.block_features[idx, self.IN_COIN_AMOUNT_IDX] += value
		self.block_features[idx, self.TX_COUNT_IDX] += 1
		self.block_features[idx, self.IN_TX_COUNT_IDX] += 1
		return True

	def _process_outTx(self, row: Any):
		outBlockID = row.outBlockID
		if pd.isna(outBlockID):
			return False
			
		value = row.value
		idx = int(outBlockID - self.minBlockID)
		
		# 更新特征数组
		self.block_features[idx, self.BLOCK_ID_IDX] = outBlockID
		self.block_features[idx, self.COIN_COUNT_IDX] += 1
		self.block_features[idx, self.COIN_AMOUNT_IDX] += value
		self.block_features[idx, self.OUT_COIN_COUNT_IDX] += 1
		self.block_features[idx, self.OUT_COIN_AMOUNT_IDX] += value
		self.block_features[idx, self.TX_COUNT_IDX] += 1
		self.block_features[idx, self.OUT_TX_COUNT_IDX] += 1
		return True
	
	def update_diff_features(self):
		"""更新差异特征（在所有块处理完成后调用）"""
		# 使用NumPy向量运算替代Python循环，速度提升显著
		self.block_features[:, self.DIF_COIN_COUNT_IDX] = (
				self.block_features[:, self.IN_COIN_COUNT_IDX] - 
				self.block_features[:, self.OUT_COIN_COUNT_IDX])
		
		self.block_features[:, self.DIF_COIN_AMOUNT_IDX] = (
				self.block_features[:, self.IN_COIN_AMOUNT_IDX] - 
				self.block_features[:, self.OUT_COIN_AMOUNT_IDX])
		
		self.block_features[:, self.DIF_TX_COUNT_IDX] = (
				self.block_features[:, self.IN_TX_COUNT_IDX] - 
				self.block_features[:, self.OUT_TX_COUNT_IDX])
