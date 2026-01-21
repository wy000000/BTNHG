# import set_parent_dir
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import copy
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from EarlyStoppingClass import EarlyStoppingClass
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from resultAnalysisClass import resultAnalysisClass
from addressTimeDataClass import addressTimeDataClass
from ExtendedNNModule import ExtendedNNModule

class DataSetModelTrainerTesterClass:
	def __init__(self, model:ExtendedNNModule,
					addressTimeDataCls:addressTimeDataClass,	
					device=None,				
					lr=BTNHGV2ParameterClass.lr,
					weight_decay=BTNHGV2ParameterClass.weight_decay,
					epochs=BTNHGV2ParameterClass.epochs,
					patience=BTNHGV2ParameterClass.patience,
					useTrainWeight=BTNHGV2ParameterClass.useTrainWeight,
					min_delta=BTNHGV2ParameterClass.min_delta,
					# 结果分析类参数
					folderPath:str=BTNHGV2ParameterClass.dataPath,
					resultFolderName:str=BTNHGV2ParameterClass.resultFolderName,
					kFold_k:int=BTNHGV2ParameterClass.kFold_k):
		
		# 检查 device 是否为 None
		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")			
		else:
			self.device = device
		print(f"using device: {self.device}")
		self._model = model
		self.addressTimeDataCls=addressTimeDataCls
		self._epochs = epochs
		self._useTrainWeight=useTrainWeight	
		self._lr = lr
		self._weight_decay=weight_decay
		self.criterion = nn.CrossEntropyLoss()


		self._trainLoader=None
		self._testLoader=None
		self._kFold_k=kFold_k
		self._trainingK=-1

		###############设置优化器#############
		self._optimizer = torch.optim.AdamW(self._model.parameters(),
										lr=self._lr,
										weight_decay=self._weight_decay)
		#####################################
		
		# # 早停相关变量
		self._min_delta=min_delta
		self._patience = patience

		###########结果分析类#############
		self.resultAnalyCls=None
		self._modelName=self._model.__class__.__name__
		self._folderPath=folderPath
		self._resultFolderName=resultFolderName

	def train_test(self, _useKFold:bool=False, **kwargs)->resultAnalysisClass:
		
		if not _useKFold:
			self._trainingK=-1
			self.resultAnalyCls=resultAnalysisClass(self._model,
					folderPath=self._folderPath,
					resultFolderName=self._resultFolderName,
					kFold_k=self._kFold_k)
			
			self._trainLoader, self._testLoader \
				= self.addressTimeDataCls.get_address_time_feature_trainLoader_testLoaser(**kwargs)
			
		# if _useKFold:
		# 	if trainLoader is None or testLoader is None:
		# 		raise ValueError("trainLoader and testLoader must be provided when _useKFold is True.")
			
		self._train()
		self._test()
		self.resultAnalyCls.compute_metrics()
		return self.resultAnalyCls

	def _train(self):
		"""
		完整训练流程
		:return: 训练历史记录
		"""
		print(f"开始训练模型，使用设备: {self.device}")

		time1 = time.time()
		resultAnalyCls=self.resultAnalyCls
		trainLoader=self._trainLoader
		# best_loss = float("inf")
		loss=float("inf")
		# counter=0
		epoch=0
		epochDisplay=BTNHGV2ParameterClass.epochsDisplay_atf
		earlyStopping=EarlyStoppingClass()
		epoch_loss_list=[]
		# trainLoader, testLoader \
		# 	= self._model.addressTimeDataCls.get_address_time_feature_trainLoader_testLoaser()

		for epoch in range(1, self._epochs + 1):
			## 训练一个 epoch
			loss, accuracy = self._train_one_epoch()

			stop=earlyStopping(val_loss=loss, accuracy=accuracy, model=self._model, epochs=epoch)

			#epoch间隔显示
			if(epoch % epochDisplay == 0 or epoch==1):
				epoch_loss_list.append((epoch, loss, accuracy))
				trainTimeStr=time.strftime('%H:%M:%S', time.gmtime(time.time() - time1))
				kFoldStr=""
				if self._trainingK!=-1:
					kFoldStr=f" | kFold: {self._trainingK}/{self._kFold_k}"

				print(self._modelName + kFoldStr
						+f" | Epoch {epoch:3d}"
						+f" | loss: {loss:.4f}"
						+f" | accuracy: {accuracy:.4f}"
						+f" | best Loss: {earlyStopping.best_loss:.4f}"
						+f" | patience: {earlyStopping.counter:2d}"
						+f" | used time: {trainTimeStr}")
				
			# 早停逻辑：监控测试集损失
			if stop:
				print(f"early stopping at epoch {epoch}.")
				break

		time2 = time.time()
		trainTimeStr=time.strftime('%H:%M:%S', time.gmtime(time2 - time1))
		resultAnalyCls.training_time=trainTimeStr

		#if epoch!=epoch_loss_list的最后一个
		if epoch!=epoch_loss_list[-1][0]:
			epoch_loss_list.append((epoch, loss, accuracy))
			
		resultAnalyCls.epoch_loss_list=epoch_loss_list
		
		endEpochLossStr=(f"Training completed, epoch : {epoch}, loss: {loss:.4f}, accuracy: {accuracy:.4f}")
		resultAnalyCls.end_epoch_loss=endEpochLossStr

		print(f"{endEpochLossStr}, used time: {trainTimeStr}")

		if earlyStopping.restore_best_weights(self._model):
			best_epoch_loss=(f"best model in epoch {earlyStopping.best_epoch},"
							+f" best loss: {earlyStopping.best_loss:.4f}"
							+f" best accuracy: {earlyStopping.best_accuracy:.4f}"
							)
			
			resultAnalyCls.trainning_accuracy=earlyStopping.best_accuracy
			resultAnalyCls.best_epoch_loss=best_epoch_loss
			#best_model_state已载入，可直接保存
			resultAnalyCls.best_model_state=copy.deepcopy(self._model.state_dict())
			print("restore "+best_epoch_loss)

			if earlyStopping.best_loss<resultAnalyCls.kFold_best_loss:				
				resultAnalyCls.kFold_best_loss=earlyStopping.best_loss
				resultAnalyCls.kFold_best_model_state=copy.deepcopy(self._model.state_dict())
		else:
			raise ValueError("Early stopping did not restore best weights.")		
		print("Training completes")
		
	def _train_one_epoch(self):
		train_dataLoader=self._trainLoader
		self._model = self._model.to(self.device)
		self._model.train()
		running_loss = 0.0
		correct = 0
		totalLabels = 0

		# train_dataLoader=self._model.train_dataLoader
		
		for batch_idx, (inputs, labels) in enumerate(train_dataLoader):
			# 移动数据到设备
			inputs, labels = inputs.to(self.device), labels.to(self.device)
			
			# 梯度清零
			self._optimizer.zero_grad()
			
			# 前向传播
			outputs = self._model(inputs)
			
			# 计算损失
			loss = self.criterion(outputs, labels)
			
			# 反向传播
			loss.backward()
			
			# 优化
			self._optimizer.step()
			
			# 统计损失和准确率
			running_loss += loss.item()
			_, predicted = outputs.max(1)
			totalLabels += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			
			# 打印训练进度			
			# if batch_idx % 1 == 0:
			# 	print(f'Batch {batch_idx}/{len(train_dataLoader)}, Loss: {loss.item():.4f}')
		
		# 计算平均损失和准确率
		avg_loss = running_loss / len(train_dataLoader)
		accuracy = correct / totalLabels
		
		return avg_loss, accuracy

	def _test(self):
		print("start test")
		time1 = time.time()
		test_dataLoader=self._testLoader

		self._model = self._model.to(self.device)
		self._model.eval()		

		all_y_true = []
		all_probs = []
		all_preds = []
		
		# test_dataLoader=self._model.test_dataLoader

		with torch.no_grad():
			for inputs, labels in test_dataLoader:
				# 移动数据到设备
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				
				# 前向传播
				logits = self._model(inputs)
				
				# 保存结果
				y_true = labels
				probs = F.softmax(logits, dim=-1)
				pred = probs.argmax(dim=-1)

				# 收集到列表
				all_y_true.append(y_true.cpu())
				all_probs.append(probs.cpu())
				all_preds.append(pred.cpu())
		# 一次性拼接
		self.resultAnalyCls.all_y_true = torch.cat(all_y_true, dim=0)
		self.resultAnalyCls.all_probs = torch.cat(all_probs, dim=0)
		self.resultAnalyCls.all_preds = torch.cat(all_preds, dim=0)

		time2 = time.time()

		#计算并输出测试accuracy
		correct = (self.resultAnalyCls.all_preds == self.resultAnalyCls.all_y_true).sum().item()
		total = len(self.resultAnalyCls.all_y_true)
		accuracy = correct / total
		print(f"测试准确率: {accuracy:.4f}")

		print(f"测试用时: {time2 - time1}")
		# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
	
	def kFold_train_test(self
						,cnn_batch_size=BTNHGV2ParameterClass.cnn_batch_size
						,shuffle=BTNHGV2ParameterClass.shuffle					
						,**kwargs)->resultAnalysisClass:
		dataSet=self.addressTimeDataCls.addressTimeFeature_dataSet
		if dataSet is None:
			raise("addressTimeFeature_dataSet is None")		
		print("start kFold_train_test")

		time1 = time.time()

		kFold_k=self._kFold_k
		self.resultAnalyCls=resultAnalysisClass(self._model,
												folderPath=self._folderPath,
												resultFolderName=self._resultFolderName,
												kFold_k=kFold_k)
		
		
		features, labels = dataSet.tensors
		KFold_indices=self.addressTimeDataCls.get_address_time_feature_KFold_indices(k=kFold_k)

		for fold_idx, (train_idx, val_idx) in enumerate(KFold_indices):
			# 显示第k折
			print(f"Processing fold {fold_idx + 1}/{kFold_k}")
			self._trainingK=fold_idx+1

			# 重置测试结果存储
			self.resultAnalyCls.all_y_true = None
			self.resultAnalyCls.all_probs = None
			self.resultAnalyCls.all_preds = None
			
			# 创建训练和验证子集
			train_dataset = TensorDataset(features[train_idx], labels[train_idx])
			test_dataset = TensorDataset(features[val_idx], labels[val_idx])
			
			# 创建DataLoader
			self._trainLoader = DataLoader(train_dataset
								, batch_size=cnn_batch_size
								, shuffle=shuffle
								)
			
			self._testLoader = DataLoader(test_dataset
								, batch_size=cnn_batch_size
								, shuffle=False
								)
			
			self._model = self._model.__class__(addressTimeFeature_dataSet=dataSet, **kwargs)
			self._optimizer = torch.optim.AdamW(self._model.parameters(),
											lr=self._lr,
											weight_decay=self._weight_decay)

			self.train_test(_useKFold=True)

			self.resultAnalyCls.kFold_evaluations.append(self.resultAnalyCls.evaluationMetrics)
			print(f"fold {fold_idx + 1}/{kFold_k} is completed.")

		self.resultAnalyCls.compute_kFold_ave_metrics()

		time2 = time.time()
		kFoldTimeStr=time.strftime('%H:%M:%S', time.gmtime(time2 - time1))
		self.resultAnalyCls.kFold_training_time=kFoldTimeStr
		print(f"kFold_train_test用时: {time2 - time1}")
		# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")
		return self.resultAnalyCls