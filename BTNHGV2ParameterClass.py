import numpy as np
import math

#定义一个参数类，保存程序中用到的各种参数
class BTNHGV2ParameterClass():
	version="25.12.1.0"
	# debugMode=False
	dataPath=r"D:\BTNHG\BTNHGV2"
	resultFolderName="result"
	train_size=0.8
	shuffle=True
	batch_size=4096
	randSeed=42
	resetSeed=False
	epochsDisplay=4
	useTrainWeight=False

	patience=8
	lr=0.01
	weight_decay=1e-4
	loss_threshold=0.0001
	stoppableLoss=0.5

	hidden_channels=32
	out_channels=32
	num_heads=4
	dropout=0.35
	num_layers=2
	HGT_useProj=True

	##################################
	epochs=512
	# epochs=16
	##################################
	# @classmethod
	# def epochs(cls, debugMode=None):
	# 	if debugMode is None:
	# 		debugMode=cls.debugMode
	# 	if debugMode:
	# 		return int(cls._epochs / 10)   # 调试模式缩短为原来的 1/10
	# 	return int(cls._epochs)

	saveBTNV2ParameterClass=True
	saveModel=False
	saveModelStateDict=True
	saveTestY=True
	saveEvaluationMetrics=True
	

	_rng = np.random.default_rng(randSeed)
	# 生成随机数
	@classmethod
	def rand(cls, isReset=resetSeed, seed=randSeed):
		if isReset:
			cls._rng = np.random.default_rng(seed)
		return int(cls._rng.integers(0, 4294967296))
	


