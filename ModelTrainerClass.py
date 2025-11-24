import torch
import torch.nn.functional as F
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
class ModelTrainerClass:
	def __init__(self, model,				
				device=None,
				lr=BTNHGV2ParameterClass.lr,
				weight_decay=BTNHGV2ParameterClass.weight_decay,
				epochs=BTNHGV2ParameterClass.epochs,
				patience=BTNHGV2ParameterClass.patience,
				loss_threshold=BTNHGV2ParameterClass.loss_threshold):
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
		self._patience = patience
		self._lr = lr
		self._weight_decay=weight_decay
		self._optimizer = torch.optim.Adam(self._model.parameters(),
										lr=self._lr,
										weight_decay=self._weight_decay)
		# # 早停相关变量
		self._loss_threshold = loss_threshold
		# self._best_acc = 0.0
		# self._best_state = None
		# self._counter = 0

	def _train_one_epoch(self):
		"""
		使用self._model, self._train_loader, self._optimizer, self._device训练一个 epoch
		Args:
			None
		Returns:
			total_loss: 该 epoch 的平均损失
		"""
		# print(next(self._model.parameters()).device)
		self._model = self._model.to(self._device)
		self._model.heteroData = self._model.heteroData.to(self._device)
		self._model.train()
		self._optimizer.zero_grad()
		out = self._model()
		outAddress = out['address']
		train_mask=self._model.train_mask
		loss = F.cross_entropy(outAddress[train_mask],\
								self._model.heteroData["address"].y[train_mask])
		loss.backward()
		self._optimizer.step()
		return loss.item()
	
		# total_loss = 0
		# for batch in self._train_loader:
		# 	batch = batch.to(self._device)
		# 	self._optimizer.zero_grad()
		# 	out = self._model()
		# 	# loss = F.cross_entropy(out, batch.y)
		# 	mask = batch.y != -1
		# 	loss = F.cross_entropy(out[mask], batch.y[mask])
		# 	loss.backward()
		# 	self._optimizer.step()
		# 	total_loss += loss.item()
		# return total_loss / len(self._train_loader)

	def run(self):
		"""完整训练与测试流程，带早停"""
		print("start train")
		best_loss = float("inf")
		loss=float("inf")
		counter=0
		epoch=0
		for epoch in range(1, self._epochs + 1):
			loss = self._train_one_epoch()
			if(epoch%10==0):
				print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
			if best_loss - loss > self._loss_threshold:
				best_loss = loss
				# best_state = self._model.state_dict()
				counter = 0
			##############################################################
			# train_acc = self.test(loader=self._train_loader)
			# test_acc = self.test(loader=self._test_loader)
			# print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
			# # 早停逻辑：监控测试集损失
			if best_loss - loss > self._loss_threshold:
				best_loss = loss
				# best_state = self._model.state_dict()
				counter = 0
			##############################################################
			#     self._best_state = self._model.state_dict()
			#     self._counter = 0
			##############################################################
			else:
				counter += 1
				if counter >= self._patience:
					print(f"早停触发！")
					break
		print(f"训练完成, epoch : {epoch}, loss: {loss:.4f}")

		# # 恢复最佳模型参数
		# if self._best_state is not None:
		#     self._model.load_state_dict(self._best_state)
		#     # #保存最佳模型参数
		#     # torch.save(self._best_state, "best_model.data")
			
	def test(self):
		"""在给定数据集上测试"""
		print("start test")
		self._model = self._model.to(self._device)
		self._model.heteroData = self._model.heteroData.to(self._device)
		# 切换到评估模式并移动到设备
		self._model.eval()
		# self._model.to(self._device)
		# print(next(self._model.parameters()).device)

		with torch.no_grad():
			# 前向传播（模型内部自己拿 self._heteroData）
			out = self._model()
			
			logits = out['address']  # shape: [num_nodes, num_classes]
			test_mask = self._model.test_mask#.to(self._device)
			y_true = self._model.heteroData["address"].y[test_mask]#.to(self._device)

			# softmax 得到置信度
			probs = F.softmax(logits[test_mask], dim=-1)
			pred = probs.argmax(dim=-1)

			# 计算准确率（分母直接用 test_mask 中 True 的数量）
			correct = (pred == y_true).sum().item()
			acc = correct / test_mask.sum().item()

			# 每个预测的置信度（取预测类别对应的概率）
			confidences = probs.max(dim=-1).values
			avg_confidence = confidences.mean().item()

			# 打印结果
			# for i, (p, conf) in enumerate(zip(pred.tolist(), confidences.tolist())):
			# 	print(f"Sample {i}: Predicted class = {p}, Confidence = {conf:.4f}")

			print(f"Accuracy: {acc:.4f}")
			print(f"Average confidence: {avg_confidence:.4f}")
		# print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")


	
