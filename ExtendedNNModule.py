import torch
import torch.nn as nn
import numpy as np
import json
from BTNHGV2ParameterClass import BTNHGV2ParameterClass

class ExtendedNNModule(nn.Module):
	def __init__(self,
				batch_size=BTNHGV2ParameterClass.batch_size,
				shuffle=BTNHGV2ParameterClass.shuffle,
				resetSeed=BTNHGV2ParameterClass.resetSeed,
				kFold_k:int=BTNHGV2ParameterClass.kFold_k):
		super().__init__()		
		self.modelName=None
		self.env_info=None
		###################
		self.training_time=None
		self.accuracy=None
		self.end_epoch_loss=None
		self.best_epoch_loss=None
		self.evaluationMetrics=None
		self.epoch_loss_list=None
		self.all_y_true = None
		self.all_probs = None
		self.all_preds = None
		####################
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.resetSeed = resetSeed
		####################
		self.kFold_k:int=kFold_k
		self.kFold_training_time=None
		self.kFold_accuracy_mean=None
		self.kFold_evaluations=[]
		self.kFold_best_loss=float('inf')
		self.kFold_best_model_state=None

	def serialize_extended_attributes(self):
		attrs = {
			"modelName": self.modelName,
			"training_time": self.training_time,
			"accuracy": self.accuracy,
			"end_epoch_loss": self.end_epoch_loss,
			"best_epoch_loss": self.best_epoch_loss,
			"evaluationMetrics": self.evaluationMetrics,
			"env_info": self.env_info
		}
		jsonStr=json.dumps(attrs, default=self._to_serializable, ensure_ascii=False, indent=4)
		# 返回 JSON 字符串
		return jsonStr
	
	def _to_serializable(self, val):
		# 处理 numpy 和 torch 类型
		if isinstance(val, (np.generic,)):
			return val.item()
		if isinstance(val, torch.Tensor):
			return val.tolist()
		if hasattr(val, "item"):  # torch scalar
			return val.item()
		return str(val)  # 兜底转换为字符串
	
	def serialize_extended_attributes_kFold(self):
		attrs = {
			"modelName": self.modelName,
			"kFold_k": self.kFold_k,
			"kFold_training_time": self.kFold_training_time,
			"kFold_accuracy_mean": self.kFold_accuracy_mean,
			"env_info": self.env_info
		}		
		jsonStr=json.dumps(attrs, default=self._to_serializable, ensure_ascii=False, indent=4)
		# 返回 JSON 字符串
		return jsonStr


	


	
