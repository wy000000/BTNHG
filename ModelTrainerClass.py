import torch
import torch.nn.functional as F
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from torch_geometric.loader import NeighborLoader

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
			# outAddress = out['address']

			# 注意：batch 中的 train_mask 是局部的，需要用 batch.train_mask
			train_mask = batch['address'].train_mask
			if train_mask.sum() == 0:  #无训练样本
				continue

			pred = outAddress[train_mask]
			target = batch['address'].y[train_mask].to(torch.long)
			#打印pred, target
			print(f"pred: {pred}")
			print(f"target: {target}")
			assert target.min() >= 0
			assert target.max() < pred.size(1)
			loss = F.cross_entropy(pred, target)

			# loss = F.cross_entropy(outAddress[train_mask], batch['address'].y[train_mask])

			loss.backward()
			self._optimizer.step()

			total_loss += loss.item()
			total_batches += 1
			##############################################
			# batch = batch.to(self._device)

			# self._optimizer.zero_grad()
			# out = self._model(batch)  # 模型前向传播
			# outAddress = out['address']  # [num_nodes_in_batch, num_classes]

			# # 注意：batch 中的 train_mask 是局部的，需要用 batch['address'].train_mask
			# train_mask = batch['address'].train_mask  # [num_nodes_in_batch]，布尔掩码

			# # 用布尔掩码索引行，得到 [num_train_nodes, num_classes]
			# pred = outAddress[train_mask]

			# # 同样用布尔掩码取标签，得到 [num_train_nodes]
			# target = batch['address'].y[train_mask]

			# loss = F.cross_entropy(pred, target)

			# loss.backward()
			# self._optimizer.step()

			# total_loss += loss.item()
			# total_batches += 1
			#######################################

		total_loss /= total_batches #if total_batches > 0 else 0

		# 返回平均损失
		return total_loss
	
		#region
		#######################################################
		# self._model = self._model.to(self._device)		
		# self._model.train()

		# # self._model.heteroData = self._model.heteroData.to(self._device)

		# self._optimizer.zero_grad()
		# out = self._model()
		# outAddress = out['address']
		# train_mask=self._model.train_mask
		# loss = F.cross_entropy(outAddress[train_mask],\
		# 						self._model.heteroData["address"].y[train_mask])
		# loss.backward()
		# self._optimizer.step()
		# return loss.item()

		#######################################################
		#endregion

		#region

		# # 梯度累积参数
		# accumulation_steps = self._model.accumulation_steps  # 根据需要调整
		
		# # 将数据分成更小的部分
		# num_chunks = accumulation_steps
		# train_mask = self._model.train_mask
		# # 计算每个chunk的样本数
		# chunk_size = (train_mask.sum().item() + num_chunks - 1) // num_chunks
		
		# total_loss = 0
		# self._optimizer.zero_grad()
		
		# for i in range(num_chunks):
		# 	# 创建当前chunk的掩码
		# 	start_idx = i * chunk_size
		# 	end_idx = min((i + 1) * chunk_size, train_mask.sum().item())
		# 	chunk_indices = torch.where(train_mask)[0][start_idx:end_idx]
			
		# 	# 创建chunk掩码
		# 	chunk_mask = torch.zeros_like(train_mask)
		# 	chunk_mask[chunk_indices] = True
			
		# 	# 前向传播和损失计算
		# 	out = self._model()
		# 	outAddress = out['address']
		# 	loss = F.cross_entropy(outAddress[chunk_mask], 
		# 						self._model.heteroData["address"].y[chunk_mask])
		# 	loss = loss / accumulation_steps  # 缩放损失
		# 	loss.backward()
		# 	total_loss += loss.item() * accumulation_steps
			
		# 	# 每accumulation_steps步更新一次参数
		# 	if (i + 1) % accumulation_steps == 0:
		# 		self._optimizer.step()
		# 		self._optimizer.zero_grad()
		
		# return total_loss
		#endregion
		
		
	################################################
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
		"""使用 neighborLoader 在测试集上测试"""
		print("start test")
		self._model = self._model.to(self._device)
		self._model.eval()

		total_correct = 0
		total_samples = 0
		total_confidence = 0.0
		total_batches = 0
		input_nodes = self._model.heteroData['address'].test_mask.nonzero(as_tuple=True)[0]
		input_nodes = input_nodes.to(torch.long).contiguous()
		nbLoader=NeighborLoader(data=self._model.heteroData,
								num_neighbors=[-1, -1],
								input_nodes=('address', input_nodes),
								batch_size=self._model.batch_size,
								shuffle=False
								)

		with torch.no_grad():
			for batch in nbLoader:
				batch = batch.to(self._device)

				# 前向传播
				outAddress = self._model(batch)
				logits = outAddress

				# 取 batch 内的 test_mask
				test_mask = batch['address'].test_mask
				if test_mask.sum().item() == 0:
					continue  # 如果该 batch 没有测试节点，跳过

				y_true = batch['address'].y[test_mask]

				# softmax 得到置信度
				probs = F.softmax(logits[test_mask], dim=-1)
				pred = probs.argmax(dim=-1)

				# 计算准确率
				correct = (pred == y_true).sum().item()
				total_correct += correct
				total_samples += test_mask.sum().item()

				# 计算平均置信度
				confidences = probs.max(dim=-1).values
				total_confidence += confidences.mean().item()
				total_batches += 1

		acc = total_correct / total_samples if total_samples > 0 else 0
		avg_confidence = total_confidence / total_batches if total_batches > 0 else 0

		print(f"Accuracy: {acc:.4f}")
		print(f"Average confidence: {avg_confidence:.4f}")
		return acc, avg_confidence




















		# ##########################################
		# print("start test")
		# self._model = self._model.to(self._device)
		
		# # 切换到评估模式并移动到设备
		# self._model.eval()
		# # self._model.to(self._device)
		# # print(next(self._model.parameters()).device)

		# with torch.no_grad():
		# 	# 前向传播（模型内部自己拿 self._heteroData）
		# 	out = self._model(self._model.heteroData)
			
		# 	logits = out['address']  # shape: [num_nodes, num_classes]
		# 	test_mask = self._model.test_mask#.to(self._device)
		# 	y_true = self._model.heteroData["address"].y[test_mask]#.to(self._device)

		# 	# softmax 得到置信度
		# 	probs = F.softmax(logits[test_mask], dim=-1)
		# 	pred = probs.argmax(dim=-1)

		# 	# 计算准确率（分母直接用 test_mask 中 True 的数量）
		# 	correct = (pred == y_true).sum().item()
		# 	acc = correct / test_mask.sum().item()

		# 	# 每个预测的置信度（取预测类别对应的概率）
		# 	confidences = probs.max(dim=-1).values
		# 	avg_confidence = confidences.mean().item()

		# 	print(f"Accuracy: {acc:.4f}")
		# 	print(f"Average confidence: {avg_confidence:.4f}")
		# # print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")