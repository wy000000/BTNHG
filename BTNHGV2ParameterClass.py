import numpy as np
import math

#定义一个参数类，保存程序中用到的各种参数
class BTNHGV2ParameterClass():
	version="2025.12.5.2"
	########### 训练
	epochs=512
	epochsDisplay=4
	useTrainWeight=False
	########### 模型
	lr=0.01
	weight_decay=1e-4
	hidden_channels=32
	out_channels=32
	num_heads=4
	dropout=0.35
	num_layers=2
	HGT_useProj=True
	########### 数据集
	train_size=0.8
	shuffle=True
	batch_size=4096
	randSeed=42
	resetSeed=False
	########### 早停
	earlyStoppingPatience=32
	min_delta=0.001
	stopableEpoch=128
	########### 存储
	dataPath=r"D:\BTNHG\BTNHGV2"
	resultFolderName="result"
	extendedAttributesFileName="extendedAttributes.txt"
	y_true_preds_probsFileName="y_true_preds_probs.xlsx"
	modelStateDictFileName="model.state_dict.pt"
	fullModelFileName="fullModel.pt"
	save=True
	saveModelStateDict=True
	saveFullModel=False	
	########### RNG
	_rng = np.random.default_rng(randSeed)
	@classmethod
	def rand(cls, isReset=resetSeed, seed=randSeed):
		if isReset:
			cls._rng = np.random.default_rng(seed)
		return int(cls._rng.integers(0, 4294967296))
	


