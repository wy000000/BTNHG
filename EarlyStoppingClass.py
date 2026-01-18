from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from resultAnalysisClass import resultAnalysisClass
from ExtendedNNModule import ExtendedNNModule
import copy

class EarlyStoppingClass:
	def __init__(self,
				patience=BTNHGV2ParameterClass.patience,
				min_delta=BTNHGV2ParameterClass.min_delta,
				stopableEpoch=BTNHGV2ParameterClass.stopableEpoch):
		self.patience = patience
		self.min_delta = min_delta
		self.stopableEpoch=stopableEpoch
		self.best_loss = None
		self.best_accuracy=None
		self.best_epoch = 0
		self.counter = 0
		self.early_stop = False
		self.best_model_state = None  # 内存保存

	def __call__(self, val_loss, accuracy, model, epochs):
		if self.best_loss is None:
			self.best_loss = val_loss
			self.best_accuracy=accuracy
			self.best_epoch=epochs
			self.best_model_state = copy.deepcopy(model.state_dict())
		elif val_loss < self.best_loss - self.min_delta:
			self.best_loss = val_loss
			self.best_accuracy=accuracy
			self.best_epoch=epochs
			self.best_model_state = copy.deepcopy(model.state_dict())
			self.counter = 0
		else:
			self.counter += 1
			if self.counter >= self.patience and epochs>=self.stopableEpoch:
				self.early_stop = True
		return self.early_stop

	def restore_best_weights(self, model:ExtendedNNModule):
		if self.best_model_state is not None:
			model.load_state_dict(self.best_model_state)
			return True
		return False

