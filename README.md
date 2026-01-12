# BTNHG
Bitcoin Transaction Network Heterogeneous Graph

version="2026.1.12.1"

fix some bugs.

version="2016.1.11"

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

six files can be saved:

	"BTNHGV2ParameterClass.txt"(parameters' setting),

	"extendedAttributes.txt"(various analysis metrics and environment configuration),

	"epoch_loss_list.xlsx"(epoch loss curve),

	"y_true_preds_probs.xlsx"(model test outputs),

	"model.state_dict.pt",

	"fullModel.pt".

sample code:

	heteroDataClass=BTNHGV2HeteroDataClass()

	# gmodel=HANClass(heteroDataCls=heteroDataClass)
	# gmodel=HGTClass(heteroDataCls=heteroDataClass)
	# gmodel=RGCNClass(heteroDataCls=heteroDataClass)
	gmodel=SAGEClass(heteroDataCls=heteroDataClass)
	# gmodel=GATClass(heteroDataCls=heteroDataClass)
	# gmodel=GraphConvClass(heteroDataCls=heteroDataClass)

	trainer=ModelTrainerTesterClass(model=gmodel)
	trainer.train()
	trainer.test()
	resultCls=resultAnalysisClass(gmodel)
	# resultCls.showEvaluationMetrics()
	# resultCls.showExtendedAttributes()
	# resultCls.plot_true_pred_counts()
	# resultCls.plot_confusion_matrix()
	resultCls.save()

	############## kFold cross-validation ##############
	trainertester=ModelTrainerTesterClass(model=gmodel)
	trainertester.kFold_train_test()
	resultCls=resultAnalysisClass(model=gmodel, kFold=True)
	resultCls.save_kFold()

