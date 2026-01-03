import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from EarlyStoppingClass import EarlyStopping

class DataSetModelTrainerTesterClass2:
	"""
	数据集模型训练测试类 - 用于训练和测试基于地址时间特征的2DCNN模型
	"""
	def __init__(self, model, train_loader, val_loader, test_loader, 
				 params, device='cuda' if torch.cuda.is_available() else 'cpu'):
		"""
		初始化方法
		:param model: 要训练的模型实例（如simple2DCNN）
		:param train_loader: 训练数据加载器
		:param val_loader: 验证数据加载器
		:param test_loader: 测试数据加载器
		:param params: 训练参数配置
		:param device: 训练设备（默认GPU，如果可用）
		"""
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.params = params
		self.device = device
		
		# 损失函数 - 二分类使用交叉熵
		self.criterion = nn.CrossEntropyLoss()
		
		# 优化器
		self.optimizer = optim.Adam(
			model.parameters(),
			lr=params.lr,
			weight_decay=params.weight_decay
		)
		
		# 早停机制
		self.early_stopping = EarlyStopping(
			patience=params.patience,
			verbose=True,
			delta=params.delta,
			path=os.path.join(params.model_save_dir, 'best_model.pt')
		)
		
		# 训练历史记录
		self.train_loss_history = []
		self.val_loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []
		self.best_accuracy = 0.0
		
	def train_epoch(self):
		"""
		训练一个epoch
		:return: 平均训练损失, 训练准确率
		"""
		self.model.train()
		running_loss = 0.0
		correct = 0
		total = 0
		
		for batch_idx, (inputs, labels) in enumerate(self.train_loader):
			# 移动数据到设备
			inputs, labels = inputs.to(self.device), labels.to(self.device)
			
			# 梯度清零
			self.optimizer.zero_grad()
			
			# 前向传播
			outputs = self.model(inputs)
			
			# 计算损失
			loss = self.criterion(outputs, labels)
			
			# 反向传播
			loss.backward()
			
			# 优化
			self.optimizer.step()
			
			# 统计损失和准确率
			running_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			
			# 打印训练进度
			if batch_idx % 10 == 0:
				print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
		
		# 计算平均损失和准确率
		avg_loss = running_loss / len(self.train_loader)
		accuracy = 100. * correct / total
		
		return avg_loss, accuracy
	
	def validate_epoch(self):
		"""
		验证一个epoch
		:return: 平均验证损失, 验证准确率
		"""
		self.model.eval()
		running_loss = 0.0
		correct = 0
		total = 0
		
		with torch.no_grad():  # 不计算梯度
			for inputs, labels in self.val_loader:
				# 移动数据到设备
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				
				# 前向传播
				outputs = self.model(inputs)
				
				# 计算损失
				loss = self.criterion(outputs, labels)
				
				# 统计损失和准确率
				running_loss += loss.item()
				_, predicted = outputs.max(1)
				total += labels.size(0)
				correct += predicted.eq(labels).sum().item()
		
		# 计算平均损失和准确率
		avg_loss = running_loss / len(self.val_loader)
		accuracy = 100. * correct / total
		
		return avg_loss, accuracy
	
	def train(self):
		"""
		完整训练流程
		:return: 训练历史记录
		"""
		print(f"开始训练模型，使用设备: {self.device}")
		start_time = time.time()
		
		for epoch in range(self.params.epochs):
			print(f"\nEpoch [{epoch+1}/{self.params.epochs}]")
			
			# 训练一个epoch
			train_loss, train_acc = self.train_epoch()
			
			# 验证一个epoch
			val_loss, val_acc = self.validate_epoch()
			
			# 记录历史数据
			self.train_loss_history.append(train_loss)
			self.val_loss_history.append(val_loss)
			self.train_acc_history.append(train_acc)
			self.val_acc_history.append(val_acc)
			
			# 打印 epoch 结果
			print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
			print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
			
			# 更新最佳准确率
			if val_acc > self.best_accuracy:
				self.best_accuracy = val_acc
				# 保存最佳模型
				if self.params.save_model:
					torch.save(self.model.state_dict(), 
							   os.path.join(self.params.model_save_dir, f'best_model_epoch_{epoch+1}.pt'))
			
			# 早停检查
			self.early_stopping(val_loss, self.model)
			if self.early_stopping.early_stop:
				print("早停触发，停止训练！")
				break
		
		# 加载最佳模型权重
		self.model.load_state_dict(torch.load(self.early_stopping.path))
		
		# 计算总训练时间
		total_time = time.time() - start_time
		print(f"\n训练完成！总耗时: {total_time:.2f}秒")
		print(f"最佳验证准确率: {self.best_accuracy:.2f}%")
		
		return {
			'train_loss': self.train_loss_history,
			'val_loss': self.val_loss_history,
			'train_acc': self.train_acc_history,
			'val_acc': self.val_acc_history,
			'best_accuracy': self.best_accuracy,
			'total_time': total_time
		}
	
	def test(self):
		"""
		测试模型性能
		:return: 测试损失, 测试准确率, 预测结果
		"""
		print("\n开始测试模型...")
		self.model.eval()
		running_loss = 0.0
		correct = 0
		total = 0
		
		all_labels = []
		all_predictions = []
		all_probabilities = []
		
		with torch.no_grad():
			for inputs, labels in self.test_loader:
				# 移动数据到设备
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				
				# 前向传播
				outputs = self.model(inputs)
				
				# 计算损失
				loss = self.criterion(outputs, labels)
				
				# 统计损失和准确率
				running_loss += loss.item()
				_, predicted = outputs.max(1)
				total += labels.size(0)
				correct += predicted.eq(labels).sum().item()
				
				# 保存结果
				all_labels.extend(labels.cpu().numpy())
				all_predictions.extend(predicted.cpu().numpy())
				all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
		
		# 计算测试结果
		test_loss = running_loss / len(self.test_loader)
		test_acc = 100. * correct / total
		
		print(f"测试完成！")
		print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
		
		return {
			'test_loss': test_loss,
			'test_acc': test_acc,
			'labels': np.array(all_labels),
			'predictions': np.array(all_predictions),
			'probabilities': np.array(all_probabilities)
		}
	
	def save_results(self, results, filename='training_results.npz'):
		"""
		保存训练和测试结果
		:param results: 包含训练和测试结果的字典
		:param filename: 保存文件名
		"""
		save_path = os.path.join(self.params.result_save_dir, filename)
		np.savez(save_path, **results)
		print(f"结果已保存到: {save_path}")
	
	def load_model(self, model_path):
		"""
		加载预训练模型权重
		:param model_path: 模型权重文件路径
		"""
		self.model.load_state_dict(torch.load(model_path))
		self.model.eval()
		print(f"模型已加载: {model_path}")