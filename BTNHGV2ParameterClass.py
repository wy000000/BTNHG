import numpy as np

#定义一个参数类，保存程序中用到的各种参数
class BTNHGV2ParameterClass():
	version="2026.1.26.1"
	########### 训练
	epochs=512 #Recommended 512
	epochsDisplay=4

	#heteroData,还未精调。当lr=0.01, useLrScheduler=False时, HGT出现过准确率最高值0.44+。
	#addressTimeData(ATD), lr:0.01-0.02, useLrScheduler=False, 易冲高，但容易出现某折低。
	#ATD, lr:0.05, useLrScheduler=True, 基本等效上面，可能稍微稳定一点。
	#ATD, lr=0.05, useLrScheduler=True, dropout=0.1, patience=128时，CNN1D_DW_SE_PE_TF 准确率0.56+。
	#ATD, 使用CNN1D_DW_SE_PE_TF_CLS，建议ropout=0.0， 准确率0.5+。
	lr=0.05
	useLrScheduler=True
	dropout=0.1
	weight_decay=0.0001
	useTrainWeight=False #效果好像不太好，建议False。

	########### 交叉验证
	kFold_k:int=5 #Recommended 5

	########### 早停
	patience=128
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

	#采用压缩数据，大量节省计算资源，建议压缩。
	compress_dataSet=True

	#是否尝试读取保存addressTimeFeature_dataSet
	########注意：测试数据预处理时设置为False，或手动删除生成的数据集##############################
	try_read_save_addressTimeFeature_dataSet=True

	#是否根据压缩前的0特征行对压缩后的数据进行部分填充。
	#false删除所有0特征行；true填充部分0特征行。特征行约相差4倍。
	#compress_padding不建议开启True，开启会增加序列长度，训练时间增加，但准确率在CNN1D_DW_SE_PE_TF没见提升。
	compress_padding=False #建议False。

	#是否对padding 0 添加轻微噪音扰动。只有在compress_padding=True时才有效。
	noisy_0=True

	#是否对addressTimeFeature中的amount类数值进行log变换，避免值过大值。
	log_addressTimeFeature_amount=True

	cnn_hidden_channels=1 #recommended 1
	cnn_out_channels=1 #recommended 1
	# cnn_hidden_fc_out=4 #recommended 4

	cnn_kernel_height=3
	cnn_kernel_width=3
	
	pool_height=2
	pool_width=1

	#########transformer
	tf_dim_feedforward=32 #减少，准确率会下降。增加，好像没什么变化。。
	tf_num_heads=1 #head增加，准确率反而下降。
	tf_num_layers=1 #layer增加，准确率未见提升，甚至会下降。