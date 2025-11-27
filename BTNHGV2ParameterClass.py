import numpy as np

#定义一个参数类，使用静态变量来保存程序中用到的各种参数
class BTNHGV2ParameterClass():
	# 定义参数
	dataPath=r"D:\BTNHG\BTNHGV2"
	train_size=0.8
	shuffle=True
	batch_size=4096
	randSeed=42
	isResetSeed=False
	epochsDisplay=4

	useTrainWeight=False

	patience=8
	lr=0.01
	weight_decay=1e-4
	epochs=512
	loss_threshold=0.0001
	stoppableLoss=0.5


	_rng = np.random.default_rng(randSeed)
	# 生成随机数
	@classmethod
	def rand(cls, isReset=isResetSeed, seed=randSeed):
		if isReset:
			cls._rng = np.random.default_rng(seed)
		return int(cls._rng.integers(0, 4294967296))
	

	#for HAN
	hidden_channels=32
	out_channels=32
	num_heads=4
	dropout=0.35
