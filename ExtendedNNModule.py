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
		self.kFold_evaluations=[]

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
	
	def serialize_extended_attributes_kFold(self, kFold:bool):
		attrs = {"modelName": self.modelName}
		if not kFold:
			attrs["training_time"] = self.training_time
			attrs["accuracy"] = self.accuracy
			attrs["end_epoch_loss"] = self.end_epoch_loss
			attrs["best_epoch_loss"] = self.best_epoch_loss
			attrs["evaluationMetrics"] = self.evaluationMetrics
		if kFold:
			attrs["kFold_k"] = self.kFold_k
			attrs["kFold_evaluations"] = self.kFold_evaluations



		attrs["env_info"] = self.env_info


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



	
