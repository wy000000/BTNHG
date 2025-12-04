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

class ModelTrainerTesterClass:
	def __init__(self, model,				
				device=None,				
				lr=BTNHGV2ParameterClass.lr,
				weight_decay=BTNHGV2ParameterClass.weight_decay,
				epochs=BTNHGV2ParameterClass.epochs,
				patience=BTNHGV2ParameterClass.patience,
				loss_threshold=BTNHGV2ParameterClass.loss_threshold,
				stoppableLoss=BTNHGV2ParameterClass.stoppableLoss,
				useTrainWeight=BTNHGV2ParameterClass.useTrainWeight):
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
		# self._model.heteroData = self._model.heteroData.to(self._device)
		# self._train_loader = train_loader
		# self._test_loader = test_loader
		self._epochs = epochs
		self._useTrainWeight=useTrainWeight

		self._patience = patience
		self._lr = lr
		self._weight_decay=weight_decay

		###############设置优化器#############
		self._optimizer = torch.optim.Adam(self._model.parameters(),
										lr=self._lr,
										weight_decay=self._weight_decay)
		#####################################
		
		# # 早停相关变量
		self._loss_threshold = loss_threshold
		self._stoppableLoss=stoppableLoss
		# self._best_acc = 0.0
		# self._best_state = None
		# self._counter = 0

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
			#print pred, target
			# print(f"pred: {pred}")
			# print(f"target: {target}")
			if(self._useTrainWeight):
				trainWeight=self._model.heteroDataCls.class_weight.to(self._device)
			else:
				trainWeight=None
			loss = F.cross_entropy(input=pred, target=target, weight=trainWeight)
			loss.backward()
			self._optimizer.step()

			total_loss += loss.item()
			total_batches += 1

		total_loss /= total_batches #if total_batches > 0 else 0

		# 返回平均损失
		return total_loss		
		
		
	################################################
	def train(self):
		"""完整训练与测试流程，带早停"""
		print("start train")
		time1 = time.time()
		best_loss = float("inf")
		loss=float("inf")
		counter=0
		epoch=0
		epochDisplay=BTNHGV2ParameterClass.epochsDisplay
		for epoch in range(1, self._epochs + 1):			
			loss = self._train_one_epoch()
			if(epoch % epochDisplay == 0):
				print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
			# if best_loss - loss > self._loss_threshold:
			# 	best_loss = loss
			# 	# best_state = self._model.state_dict()
			# 	counter = 0
			##############################################################
			# train_acc = self.test(loader=self._train_loader)
			# test_acc = self.test(loader=self._test_loader)
			# print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
			# # 早停逻辑：监控测试集损失
			if loss<best_loss:
				best_loss = loss
				# best_state = self._model.state_dict()
				counter = 0
			##############################################################
			#     self._best_state = self._model.state_dict()
			#     self._counter = 0
			##############################################################
			else:
				counter += 1

			if best_loss - loss < self._loss_threshold\
				and loss<self._stoppableLoss and epoch>self._patience\
				and counter>=self._patience:
					print(f"早停触发！")
					break
		print(f"训练完成, epoch : {epoch}, loss: {loss:.4f}")
		time2 = time.time()

		#打印训练用时，格式为时：分：秒
		trainTimeStr=time.strftime('%H:%M:%S', time.gmtime(time2 - time1))
		self._model.training_time=trainTimeStr
		print(f"训练用时: {trainTimeStr}")
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

		# # 恢复最佳模型参数
		# if self._best_state is not None:
		#     self._model.load_state_dict(self._best_state)
		#     # #保存最佳模型参数
		#     # torch.save(self._best_state, "best_model.data")

	def test(self):
		"""使用 neighborLoader 在测试集上测试，并展示混淆矩阵"""
		print("start test")
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
		self._model.all_y_true = torch.cat(all_y_true, dim=0)
		self._model.all_probs = torch.cat(all_probs, dim=0)
		self._model.all_preds = torch.cat(all_preds, dim=0)		

		time2 = time.time()
		print(f"测试用时: {time2 - time1}")
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

	
