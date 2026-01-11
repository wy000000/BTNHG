import numpy as np

#定义一个参数类，保存程序中用到的各种参数
class BTNHGV2ParameterClass():
	version="2016.1.11"
	########### 训练
	###############################	
	epochs=512
	###############################
	epochsDisplay_hetero=4
	lr=0.01
	# max_norm=100000000.0
	dropout=0.35
	weight_decay=1e-4
	useTrainWeight=False
	
	########### 交叉验证
	# useKFold:bool=False
	kFold_k:int=5

	########### 早停
	patience=32
	min_delta=0.01
	stopableEpoch=128

	########### 模型	
	hidden_channels=32
	out_channels=32
	num_heads=4	
	num_layers=2
	HGT_useProj=True

	########### 数据集
	train_size=0.8
	shuffle=True
	batch_size=4096
	randSeed=42
	resetSeed=False

	########### 存储
	dataPath=r"D:\BTNHG\BTNHGV2"
	resultFolderName="result"
	extendedAttributesFileName="extendedAttributes.txt"
	epoch_loss_listFileName="epoch_loss_list.xlsx"
	y_true_preds_probsFileName="y_true_preds_probs.xlsx"
	modelStateDictFileName="model.state_dict.pt"
	fullModelFileName="fullModel.pt"
	kFold_evaluationMetricsFileName="kFold_evaluationMetrics.xlsx"
	addressTimeFeature_dataSet_name="addressTimeFeature_dataSet.pt"
	save=True
	saveModelStateDict=True
	# saveFullModel=False
	
	########### RNG
	_rng = np.random.default_rng(randSeed)
	@classmethod
	def rand(cls, isReset=resetSeed, seed=randSeed):
		if isReset:
			cls._rng = np.random.default_rng(seed)
		return int(cls._rng.integers(0, 4294967296))
	
	################# address Time Feature ######################
	cnn_hidden_channels=2
	cnn_out_channels=2
	cnn_kernel_size=3
	pool_width=1
	pool_height=2
	cnn_batch_size=256

	epochsDisplay_atf=1


	


