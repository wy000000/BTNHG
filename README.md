# BTNHG
Bitcoin Transaction Network Heterogeneous Graph

version="2026.1.26.1"

added learning rate scheduler.

version="2026.1.20.1"

记录训练准确率，方便监测过拟合。

version="2026.1.14.1"

修复交叉验证没有重置模型的bug。

version="2025.12.18.1"

kFold cross-validation is added.

version="2025.12.6.1"

Demonstrating how to use the BTNHG2013v2 dataset in PyTorch.

dataset https://www.kaggle.com/datasets/wy000000/btnhgv2

HAN, HGT, RGCN, SAGE, GAT and GraphConv methods are integrated.

code files：

	"BTNHGV2.py" is main file.

	"BTNHGV2ParameterClass.py" stores parameters.

	"BTNHGV2HeteroDataClass.py" loads BTNHGV2 dataset to HeteroData.

	"ExtendedNNModule.py" extends nn.Module.

	"ModelTrainerTesterClass.py" trains and tests model.

	"resultAnalysisClass.py" implements result analysis.

	"EarlyStoppingClass.py" implements early stopping mechanism.

sample code:

	# 处理数据集
	heteroDataCls=BTNHGV2HeteroDataClass()

	# 定义模型
	# gmodel=HANClass(heteroData=heteroDataCls.heteroData)
	# gmodel=HGTClass(heteroData=heteroDataCls.heteroData)
	# gmodel=RGCNClass(heteroData=heteroDataCls.heteroData)
	gmodel=SAGEClass(heteroData=heteroDataCls.heteroData)
	# gmodel=GATClass(heteroData=heteroDataCls.heteroData)
	# gmodel=GraphConvClass(heteroData=heteroDataCls.heteroData)

	# 准备训练器测试器
	trainertester=HeteroModelTrainerTesterClass(model=gmodel)

	#单次训练测试
	resultAnalyCls=trainertester.train_test()
	resultAnalyCls.save()

	#辅助显示
	resultAnalyCls.showEvaluationMetrics()
	resultAnalyCls.showExtendedAttributes()
	resultAnalyCls.plot_true_pred_counts()
	resultAnalyCls.plot_confusion_matrix()

	# 交叉验证
	resultAnalyCls=trainertester.kFold_train_test()
	resultAnalyCls.save_kFold()

