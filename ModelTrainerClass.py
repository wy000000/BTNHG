import torch
import torch.nn.functional as F
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
class ModelTrainerClass:
	def __init__(self, model, train_loader, test_loader, device=None, 
				lr=BTNHGV2ParameterClass.lr,
				weight_decay=BTNHGV2ParameterClass.weight_decay,
				epochs=BTNHGV2ParameterClass.epochs,
				patience=BTNHGV2ParameterClass.patience,
				loss_threshold=BTNHGV2ParameterClass.loss_threshold):
		"""
		通用训练器，支持早停
		Args:
			model: 传入的模型 (HAN, HGT, RGCN 等)
			train_loader: 训练数据加载器
			test_loader: 测试数据加载器
			device: 运行设备 ("cuda" 或 "cpu")
			lr: 学习率
			weight_decay: 权重衰减
			epochs: 最大训练轮数
			patience: 早停容忍度 (多少个 epoch 没提升就停止)
		"""
		# 检查 device 是否为 None
		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = device
		
		self.model = model.to(self.device)
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.epochs = epochs
		self.patience = patience
		self.lr = lr
		self.weight_decay=weight_decay
		self.optimizer = torch.optim.Adam(model.parameters(),
										lr=self.lr,
										weight_decay=self.weight_decay)
		# # 早停相关变量
		self.loss_threshold = loss_threshold

		# self.best_acc = 0.0
		# self.best_state = None
		# self.counter = 0

	def train_one_epoch(self):
		"""训练一个 epoch"""
		self.model.train()
		total_loss = 0
		for batch in self.train_loader:
			batch = batch.to(self.device)
			self.optimizer.zero_grad()
			out = self.model(batch)
			loss = F.cross_entropy(out, batch.y)
			loss.backward()
			self.optimizer.step()
			total_loss += loss.item()
		return total_loss / len(self.train_loader)

	def test(self, loader):
		"""在给定数据集上测试"""
		self.model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for batch in loader:
				batch = batch.to(self.device)
				out = self.model(batch)
				pred = out.argmax(dim=-1)
				correct += int((pred == batch.y).sum())
				total += batch.y.size(0)
		print(f"Accuracy: {correct / total:.4f}")
		return correct / total

	def run(self):
		"""完整训练与测试流程，带早停"""
		best_loss = float("inf")
		loss=float("inf")
		counter=0
		epoch=0
		for epoch in range(1, self.epochs + 1):
			loss = self.train_one_epoch()
			if(epoch%10==0):
				print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
			if best_loss - loss > self.loss_threshold:
				best_loss = loss
				# best_state = self.model.state_dict()
				counter = 0
			##############################################################
			# train_acc = self.test(self.train_loader)
			# test_acc = self.test(self.test_loader)
			# print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
			# # 早停逻辑：监控测试集准确率
			# if test_acc > self.best_acc:
			#     self.best_acc = test_acc
			#     self.best_state = self.model.state_dict()
			#     self.counter = 0
			##############################################################
			else:
				counter += 1
				if counter >= self.patience:
					print(f"早停触发！")
					break
		print(f"训练完成, epoch : {epoch}, loss: {loss:.4f}")

		# # 恢复最佳模型参数
		# if self.best_state is not None:
		#     self.model.load_state_dict(self.best_state)
		#     # #保存最佳模型参数
		#     # torch.save(self.best_state, "best_model.data")
			
			
