import torch
import torch.nn as nn
import numpy as np
import json

class ExtendedNNModule(nn.Module):
	def __init__(self):
		super().__init__()		
		self.modelName=None
		self.training_time=None
		self.accuracy=None
		self.end_epoch_loss=None
		self.best_epoch_loss=None
		self.evaluationMetrics=None
		self.env_info=None
		self.epoch_loss_list=None
		self.all_y_true = None
		self.all_probs = None
		self.all_preds = None

	def serialize_extended_attributes(self):
		attrs = {
			"modelName": self.modelName,
			"training_time": self.training_time,
			"accuracy": self.accuracy,
			"end_epoch_loss": self.end_epoch_loss,
			"best_epoch_loss": self.best_epoch_loss,
			"evaluationMetrics": self.evaluationMetrics,
			"env_info": self.env_info,
		}
		def to_serializable(val):
			# 处理 numpy 和 torch 类型
			if isinstance(val, (np.generic,)):
				return val.item()
			if isinstance(val, torch.Tensor):
				return val.tolist()
			if hasattr(val, "item"):  # torch scalar
				return val.item()
			return str(val)  # 兜底转换为字符串
		jsonStr=json.dumps(attrs, default=to_serializable, ensure_ascii=False, indent=4)
		# 返回 JSON 字符串
		return jsonStr
	
