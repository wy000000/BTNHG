from BTNHGV2ParameterClass import BTNHGV2ParameterClass
import os
import datetime
from dataclasses import dataclass, asdict
import numpy as np
import shutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import platform
import psutil
import sys
import cpuinfo
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
	confusion_matrix,
	ConfusionMatrixDisplay)

class resultAnalysisClass:
	def __init__(self,
				folderPath:str=BTNHGV2ParameterClass.dataPath,
				resultFolderName:str=BTNHGV2ParameterClass.resultFolderName,
				model:nn.Module=None,
				save:bool=
				):
		self.resultFolderName=resultFolderName
		self.resultFolderPath=os.path.join(folderPath, self.resultFolderName)
		self.methodFolderName=None
		self.methodFolderPath=None
		self.model=model
		self.modelName=self.model.__class__.__name__
		self.accuracy=0.0
		self.evaluationMetrics=self._compute_metrics(
											y_true=self.model.all_y_true,
											y_preds=self.model.all_y_preds,
											y_probs=self.model.all_y_probs)
		self.evaluationMetrics["modelName"]=self.modelName
		self.evaluationMetrics["training_time"]=self.model.training_time

		#配置信息
		self.env_info=self._get_training_env_info()

		
		
		

	def showEvaluationMetrics(self):
		print(self.evaluationMetrics)

	def plot_true_pred_counts(self, y_true, y_preds):
		# 绘制预测与真实数量的柱状图
		# 获取所有类别
		classes = np.unique(np.concatenate([y_preds, y_true]))

		# 统计每个类别的预测数量和真实数量
		pred_counts = [np.sum(np.array(y_preds) == c) for c in classes]
		label_counts = [np.sum(np.array(y_true) == c) for c in classes]

		# 绘制柱状图
		x = np.arange(len(classes))  # 类别索引
		width = 0.35  # 柱子宽度

		fig, ax = plt.subplots(figsize=(8, 6))
		rects1 = ax.bar(x - width/2, pred_counts, width, label='Predicted', color='skyblue')
		rects2 = ax.bar(x + width/2, label_counts, width, label='True', color='salmon')

		# 添加标签和标题
		ax.set_xlabel('Classes')
		ax.set_ylabel('Counts')
		ax.set_title('Predicted vs True Counts per Class')
		ax.set_xticks(x)
		ax.set_xticklabels(classes)
		ax.legend()

		# 在柱子上标注数值
		for rects in [rects1, rects2]:
			for rect in rects:
				height = rect.get_height()
				ax.annotate(f'{height}',
							xy=(rect.get_x() + rect.get_width() / 2, height),
							xytext=(0, 3),  # 向上偏移 3
							textcoords="offset points",
							ha='center', va='bottom')

		plt.tight_layout()
		plt.show()
		return

	def plot_confusion_matrix(self, y_true, y_preds):
		cm = confusion_matrix(y_true, y_preds)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm)
		disp.plot(cmap=plt.cm.Blues)
		plt.title("Confusion Matrix on Test Set")
		plt.show()

	#计算评估指标
	def _compute_metrics(self, y_true, y_preds, y_probs):
		"""
		计算常用评估指标，包括类别不均衡场景下的指标:
		- Accuracy
		- Balanced Accuracy
		- Precision/Recall/F1 (macro & weighted)
		- ROC-AUC (多分类时用宏平均)
		- PR-AUC (多分类时用宏平均)
		- Cohen's Kappa
		- Matthews Correlation Coefficient (MCC)
		- 平均置信度 (max softmax 概率的平均值)
		"""

		# 转换为 numpy
		y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
		y_preds = y_preds.cpu().numpy() if isinstance(y_preds, torch.Tensor) else np.array(y_preds)
		y_probs = y_probs.cpu().numpy() if isinstance(y_probs, torch.Tensor) else np.array(y_probs)

		# 基础指标
		acc = accuracy_score(y_true, y_preds)
		self.accuracy=acc
		bal_acc = balanced_accuracy_score(y_true, y_preds)

		# Precision / Recall / F1
		prec_macro = precision_score(y_true, y_preds, average="macro", zero_division=0)
		rec_macro = recall_score(y_true, y_preds, average="macro", zero_division=0)
		f1_macro = f1_score(y_true, y_preds, average="macro", zero_division=0)

		prec_weighted = precision_score(y_true, y_preds, average="weighted", zero_division=0)
		rec_weighted = recall_score(y_true, y_preds, average="weighted", zero_division=0)
		f1_weighted = f1_score(y_true, y_preds, average="weighted", zero_division=0)

		# ROC-AUC & PR-AUC (多分类时用宏平均)
		try:
			roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
		except Exception:
			roc_auc = None

		try:
			pr_auc = average_precision_score(y_true, y_probs, average="macro")
		except Exception:
			pr_auc = None

		# Cohen's Kappa & MCC
		kappa = cohen_kappa_score(y_true, y_preds)
		mcc = matthews_corrcoef(y_true, y_preds)

		# 平均置信度
		confidences = y_probs.max(axis=1)
		avg_conf = confidences.mean()

		return {
			"accuracy": acc,
			"balanced_accuracy": bal_acc,
			"precision_macro": prec_macro,
			"recall_macro": rec_macro,
			"f1_macro": f1_macro,
			"precision_weighted": prec_weighted,
			"recall_weighted": rec_weighted,
			"f1_weighted": f1_weighted,
			"roc_auc_macro": roc_auc,
			"pr_auc_macro": pr_auc,
			"cohen_kappa": kappa,
			"mcc": mcc,
			"avg_confidence": avg_conf
		}

	def _get_training_env_info(self):
		"""
		获取神经网络训练常用的软件和硬件配置信息
		"""
		info = {}

		# 🖥️ 硬件信息
		info["System"] = platform.system()
		info["Architecture"] = cpuinfo.get_cpu_info()['arch']
		info["CPU"] = cpuinfo.get_cpu_info()['brand_raw']
		info["Cores"] = psutil.cpu_count(logical=False)
		info["memory_total_GB"] = round(psutil.virtual_memory().total / (1024**3), 2)

		# GPU 信息（PyTorch）
		if torch.cuda.is_available():
			info["gpu_available"] = True
			info["gpu_count"] = torch.cuda.device_count()

			for i in range(info["gpu_count"]):
				props = torch.cuda.get_device_properties(i)
				name = torch.cuda.get_device_name(i)
				total_mem = round(props.total_memory / (1024**3), 2)  # 转换为 GB
				info[f"GPU {i}"] = f"{name} {total_mem} GB"
		else:
			info["gpu_available"] = False

		# 📦 软件信息
		info["python_version"] = sys.version.split()[0]
		info["torch_version"] = torch.__version__

		return info

	def save(self):
		self._createMethodFolder()
		self._saveBTNHGV2ParameterClass()



		return
	
	def _createMethodFolder(self):
			modelName=self.modelName
			accuracy=self.accuracy
			#folderName=modelName+年月日时分秒+accuracy,以"-"分隔
			self.methodFolderName=modelName+"-"+datetime.datetime.now().strftime("%Y.%m.%d %H.%M.%S")\
						+"-"+f"acc {accuracy:.4f}"
			#将path与folderName拼接
			self.methodFolderPath=os.path.join(self.resultFolderPath, self.methodFolderName)
			print(f"methodFolderPath={self.methodFolderPath}")
			#创建文件夹
			os.makedirs(self.methodFolderPath, exist_ok=True)
			return self.methodFolderPath
	
	def _saveBTNHGV2ParameterClass(self):
		#复制BTNHGV2ParameterClass.py 到 folderPath
		fileName="BTNHGV2ParameterClass.py"
		try:
			shutil.copyfile(fileName, os.path.join(self.methodFolderPath, fileName))
			print(f"{fileName}已保存")
		except Exception as e:
			print(f"保存失败: {e}")
	
	def _saveModel_state_dict(self):
		#复制model.state_dict() 到 self.methodFolderPath
		fileName="model.state_dict.pt"
		try:
			torch.save(self.model.state_dict(), os.path.join(self.methodFolderPath, fileName))
			print(f"{fileName}已保存")
		except Exception as e:
			print(f"保存失败: {e}")
	def _saveFullModel(self):
		#复制model 到 self.methodFolderPath
		fileName="fullModel.pt"
		try:
			torch.save(self.model, os.path.join(self.methodFolderPath, fileName))
			print(f"{fileName}已保存")
		except Exception as e:
			print(f"保存失败: {e}")
		

	


	

