import numpy as np

#定义一个参数类，保存程序中用到的各种参数
class BTNHGV2ParameterClass():
	version="2026.1.26.1"
	########### 训练
	epochs=512 #Recommended 512
	epochsDisplay_hetero=4

	lr=0.01
	weight_decay=0.0001
	useLrScheduler=True
	dropout=0.35
	useTrainWeight=False

	########### 交叉验证
	kFold_k:int=5 #Recommended 5

	########### 早停
	patience=32 #recommended 32
	min_delta=0.01
	stopableEpoch=128 #recommended 128

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
	
	######## address Time Feature###################
	minBlockID=272375
	maxBlockID=277995

	#采用压缩数据，大量节省计算资源，但会损失准确率
	compress_dataSet=True

	#是否尝试读取保存addressTimeFeature_dataSet##################################
	#测试数据预处理时设置为False
	try_read_save_addressTimeFeature_dataSet=True

	#是否对压缩数据进行padding
	compress_padding=True
	#是否padding 0 添加轻微噪音扰动
	noisy_0=True

	#是否对addressTimeFeature中的amount类数值进行log变换，避免值过大值
	log_addressTimeFeature_amount=True

	cnn_hidden_channels=1 #recommended 1
	cnn_out_channels=1 #recommended 1
	# cnn_hidden_fc_out=4 #recommended 4

	cnn_kernel_height=3
	cnn_kernel_width=3
	
	pool_height=2
	pool_width=1

	cnn_batch_size=4096

	epochsDisplay_atf=4


	#########transformer
	tf_dim_feedforward=32
	tf_num_heads=1
	tf_num_layers=1