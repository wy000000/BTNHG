import torch
import torch.nn.functional as F
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from torch_geometric.loader import NeighborLoader
import time
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from EarlyStoppingClass import EarlyStoppingClass
import copy
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
from resultAnalysisClass import resultAnalysisClass
from ExtendedNNModule import ExtendedNNModule
from transformers import get_cosine_schedule_with_warmup

class HeteroModelTrainerTesterClass:
	def __init__(self, model:ExtendedNNModule,
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
				kFold_k:int=BTNHGV2ParameterClass.kFold_k,
				batch_size=BTNHGV2ParameterClass.batch_size,
				useLrScheduler=BTNHGV2ParameterClass.useLrScheduler,
				epochsDisplay=BTNHGV2ParameterClass.epochsDisplay
				):
		"""
		通用训练器，支持早停
		Args:
			model: 传入的模型 (HAN, HGT, RGCN 等)
			device: 运行设备 ("cuda" 或 "cpu")
			lr: 学习率
			weight_decay: 权重衰减
			epochs: 最大训练轮数
			patience: 早停容忍度 (多少个 epoch 没提升就停止)
			loss_threshold: 损失阈值 (早停触发条件)
		"""
		# 检查 device 是否为 None
		if device is None:
			self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")			
		else:
			self._device = device
		print(f"using device: {self._device}")

		self._model = model

		###########训练相关参数#############
		self._epochs = epochs
		self._epochsDisplay=epochsDisplay
		self._useTrainWeight=useTrainWeight	
		self._lr = lr
		self._useLrScheduler=useLrScheduler
		self._weight_decay=weight_decay

		############kFold相关参数#############
		# self._useKFold=useKFold
		self._kFold_k=kFold_k
		self._trainingK=-1

		###############设置优化器#############
		self._optimizer = torch.optim.AdamW(self._model.parameters(),
										lr=self._lr,
										weight_decay=self._weight_decay)
		#####################################
		
		self._batch_size=batch_size
		datasetSize = len(self._model.heteroData['address'].train_mask.nonzero(as_tuple=True)[0])
		
		# 每个 epoch 的 batch 数
		batches_per_epoch = (datasetSize + self._batch_size - 1) // self._batch_size
		# 总 batch 数
		self._total_num_batches = batches_per_epoch * self._epochs
		self._warmup_steps = int(self._total_num_batches * 0.1)

		self._lr_scheduler = get_cosine_schedule_with_warmup(
				optimizer=self._optimizer,
				num_warmup_steps=self._warmup_steps,
				num_training_steps=self._total_num_batches)
		
		###########早停相关变量#############
		self._min_delta=min_delta
		self._patience = patience		
		# self._loss_threshold = loss_threshold
		# self._stoppableLoss=stoppableLoss
		# self._best_acc = 0.0
		# self._best_state = None
		# self._counter = 0
	
		###########结果分析类#############
		self.resultAnalyCls=None
		self._modelName=self._model.__class__.__name__
		self._folderPath=folderPath
		self._resultFolderName=resultFolderName

	def train_test(self, _useKFold:bool=False)->resultAnalysisClass:
		#测试训练集通过heteroData['address'].kFold_masks传递
		if not _useKFold:
			self._trainingK=-1
			self.resultAnalyCls=resultAnalysisClass(self._model,
								folderPath=self._folderPath,
								resultFolderName=self._resultFolderName,
								# useKFold=False,
								kFold_k=self._kFold_k)
		self._train()
		self._test()
		self.resultAnalyCls.compute_metrics()
		return self.resultAnalyCls

	def _train(self):
		"""完整训练与测试流程，带早停"""
		print("Training starts")
		time1 = time.time()
		resultAnalyCls=self.resultAnalyCls
		# best_loss = float("inf")
		loss=float("inf")
		# counter=0
		epoch=0
		epochDisplay=self._epochsDisplay
		earlyStopping=EarlyStoppingClass()		
		epoch_loss_list=[]

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
		# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

	def _train_one_epoch(self):
		"""
		使用self._model, neighborLoader, self._optimizer, self._device训练一个 epoch
		Args:
			neighborLoader: 邻居加载器
		Returns:
			total_loss: 该 epoch 的平均损失
		"""
		input_nodes = self._model.heteroData['address'].train_mask.nonzero(as_tuple=True)[0]
		input_nodes = input_nodes.to(torch.long).contiguous()
		nbLoader=NeighborLoader(data=self._model.heteroData,
					num_neighbors=[-1, -1],
					input_nodes=('address', input_nodes),
					batch_size=self._model.batch_size,
					shuffle=self._model.shuffle,
					)
		self._model = self._model.to(self._device)
		self._model.train()
		total_loss = 0
		total_batches = 0
		total_correct = 0
		total_samples = 0

		# 遍历 neighborLoader 提供的批次
		for batch in nbLoader:
			batch = batch.to(self._device)

			self._optimizer.zero_grad()
			outAddress = self._model(batch)  # 模型前向传播，输入是 batch

			# 注意：batch 中的 train_mask 是局部的，需要用 batch.train_mask
			train_mask = batch['address'].train_mask
			if train_mask.sum() == 0:
				continue

			pred = outAddress[train_mask]
			target = batch['address'].y[train_mask].to(torch.long)

			if(self._useTrainWeight):
				trainWeight=self._model.heteroDataCls.class_weight.to(self._device)
			else:
				trainWeight=None
			loss = F.cross_entropy(input=pred, target=target, weight=trainWeight)
			loss.backward()

			self._optimizer.step()
			# 更新学习率
			if self._useLrScheduler:
				self._lr_scheduler.step()

			total_loss += loss.item()
			total_batches += 1

			# ====== 计算准确率 ======
			predicted_classes = pred.argmax(dim=1) # 取最大概率对应的类别
			correct = (predicted_classes == target).sum().item()
			total_correct += correct
			total_samples += target.size(0)

		total_loss /= total_batches #if total_batches > 0 else 0

		# 总体准确率
		accuracy = total_correct / total_samples# if total_samples > 0 else 0.0		

		# 返回平均损失
		return total_loss, accuracy

	def _test(self):
		"""使用 neighborLoader 在测试集上测试"""
		print("Testing starts")
		time1 = time.time()
		self._model = self._model.to(self._device)
		self._model.eval()		

		all_preds = []

		input_nodes = self._model.heteroData['address'].test_mask.nonzero(as_tuple=True)[0]
		input_nodes = input_nodes.to(torch.long).contiguous()

		nbLoader = NeighborLoader(
			data=self._model.heteroData,
			num_neighbors=[-1, -1],
			input_nodes=('address', input_nodes),
			batch_size=self._model.batch_size,
			shuffle=False
		)

		all_y_true = []
		all_probs = []
		all_preds = []

		with torch.no_grad():
			for batch in nbLoader:
				batch = batch.to(self._device)
				logits = self._model(batch)
				test_mask = batch['address'].test_mask

				if test_mask.sum().item() == 0:
					continue

				y_true = batch['address'].y[test_mask]
				probs = F.softmax(logits[test_mask], dim=-1)
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

		print(f"Testing completes, 测试用时: {time2 - time1}")
		# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

	def kFold_train_test(self, **kwargs)->resultAnalysisClass:
		print("start kFold_train_test")
		time1 = time.time()		
		self.resultAnalyCls=resultAnalysisClass(self._model,
					folderPath=self._folderPath,
					resultFolderName=self._resultFolderName,
					kFold_k=self._kFold_k)
					# _useKFold=True)

		# heteroDataCls=self._model.heteroDataCls
		heteroData=self._model.heteroData

		k=1
		# 进行 k 折交叉验证
		for train_mask, test_mask in heteroData['address'].kFold_masks:
			print(f"{k} Fold, total {self._kFold_k} fold")
			self._trainingK=k
			heteroData['address'].train_mask=train_mask
			heteroData['address'].test_mask=test_mask
			
			# 重置测试结果存储
			self.resultAnalyCls.all_y_true = None
			self.resultAnalyCls.all_probs = None
			self.resultAnalyCls.all_preds = None
			
			# 初始化模型和优化器
			self._model=self._model.__class__(heteroData=heteroData, **kwargs)
			self._optimizer = torch.optim.AdamW(self._model.parameters(),
											lr=self._lr,
											weight_decay=self._weight_decay)
			self._lr_scheduler = get_cosine_schedule_with_warmup(
				optimizer=self._optimizer,
				num_warmup_steps=self._warmup_steps,
				num_training_steps=self._total_num_batches)
			
			self.train_test(_useKFold=True)
			self.resultAnalyCls.kFold_evaluations.append(self.resultAnalyCls.evaluationMetrics)
			print(f"fold {k}/{self._kFold_k} is completed.")
			k+=1
		
		self.resultAnalyCls.compute_kFold_ave_metrics()

		time2 = time.time()
		kFoldTimeStr=time.strftime('%H:%M:%S', time.gmtime(time2 - time1))
		self.resultAnalyCls.kFold_training_time=kFoldTimeStr
		print(f"kFold_train_test用时: {time2 - time1}")
		# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

		return self.resultAnalyCls