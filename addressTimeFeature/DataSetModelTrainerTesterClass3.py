import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import os


class DataSetModelTrainerTesterClass3:
    """
    用于训练和测试基于TensorDataset的模型的通用类
    """

    def __init__(self, model: nn.Module, train_dataset: TensorDataset, 
                 test_dataset: TensorDataset, validation_dataset: TensorDataset = None,
                 batch_size: int = 32, learning_rate: float = 0.001,
                 num_epochs: int = 100, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器测试器
        
        Args:
            model: 要训练的神经网络模型
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            validation_dataset: 验证数据集（可选）
            batch_size: 批次大小
            learning_rate: 学习率
            num_epochs: 训练轮数
            device: 训练设备
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device

        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False) if validation_dataset else None

        # 优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史记录
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Returns:
            (平均损失, 平均准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device).long()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        验证模型
        
        Returns:
            (平均损失, 平均准确率)
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device).long()
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def test(self) -> Dict[str, float]:
        """
        测试模型
        
        Returns:
            包含各种评估指标的字典
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device).long()
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets)

        # 计算各种指标
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            verbose: 是否打印训练进度
            
        Returns:
            训练历史和最终测试结果
        """
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}] '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 最终测试
        test_results = self.test()

        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'test_results': test_results
        }

    def plot_training_history(self):
        """
        绘制训练历史
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失图
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # 准确率图
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, path)

    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型文件路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']