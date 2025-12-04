# BTNHG
Bitcoin Transaction Network Heterogeneous Graph

Demonstrating how to use the BTNHG2013v2 dataset in PyTorch.

BTNHGV2.py is main file.

BTNHGV2ParameterClass.py stores parameters.

code:

heteroDataClass=BTNHGV2HeteroDataClass()

gmodel=myNN(heteroDataCls=heteroDataClass) \# define your nn in myNN.

trainer=ModelTrainerTesterClass(model=gmodel)

trainer.train()

trainer.test()

resultCls=resultAnalysisClass(gmodel)

resultCls.showEvaluationMetrics()

resultCls.showExtendedAttributes()

resultCls.plot_true_pred_counts()

resultCls.plot_confusion_matrix()

resultCls.save() \# save BTNHGV2ParameterClass.py, extendedAttributes.txt, y_true_preds_probs.xlsx, model.state_dict.pt, fullModel.pt



