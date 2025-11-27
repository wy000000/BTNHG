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

class ModelTrainerClass:
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
		self._optimizer = torch.optim.Adam(self._model.parameters(),
										lr=self._lr,
										weight_decay=self._weight_decay)
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
	def run(self):
		"""完整训练与测试流程，带早停"""
		print("start train")
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

		# # 恢复最佳模型参数
		# if self._best_state is not None:
		#     self._model.load_state_dict(self._best_state)
		#     # #保存最佳模型参数
		#     # torch.save(self._best_state, "best_model.data")
			


	def test(self):
		"""使用 neighborLoader 在测试集上测试，并展示混淆矩阵"""
		print("start test")
		self._model = self._model.to(self._device)
		self._model.eval()

		total_correct = 0
		total_samples = 0
		total_confidence = 0.0
		total_batches = 0

		all_preds = []
		all_labels = []

		input_nodes = self._model.heteroData['address'].test_mask.nonzero(as_tuple=True)[0]
		input_nodes = input_nodes.to(torch.long).contiguous()

		nbLoader = NeighborLoader(
			data=self._model.heteroData,
			num_neighbors=[-1, -1],
			input_nodes=('address', input_nodes),
			batch_size=self._model.batch_size,
			shuffle=False
		)

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

				# 累积准确率和置信度
				correct = (pred == y_true).sum().item()
				total_correct += correct
				total_samples += test_mask.sum().item()
				confidences = probs.max(dim=-1).values
				total_confidence += confidences.mean().item()
				total_batches += 1

				# 收集所有预测和真实标签
				all_preds.extend(pred.cpu().numpy())
				all_labels.extend(y_true.cpu().numpy())

		acc = total_correct / total_samples if total_samples > 0 else 0
		avg_confidence = total_confidence / total_batches if total_batches > 0 else 0
		balanced_acc = balanced_accuracy_score(all_labels, all_preds)		

		print(f"Accuracy: {acc:.4f}")
		print(f"Average confidence: {avg_confidence:.4f}")
		print(f"Balanced Accuracy: {balanced_acc:.4f}")
		self._model.heteroDataCls.printClusterCount()
		print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")


		# 绘制混淆矩阵
		cm = confusion_matrix(all_labels, all_preds)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm)
		disp.plot(cmap=plt.cm.Blues)
		plt.title("Confusion Matrix on Test Set")
		plt.show()

		return acc, avg_confidence
