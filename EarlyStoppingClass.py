from BTNHGV2ParameterClass import BTNHGV2ParameterClass

import copy

class EarlyStoppingClass:
	def __init__(self,
				patience=BTNHGV2ParameterClass.earlyStoppingPatience,
				min_delta=BTNHGV2ParameterClass.min_delta,
				stopableEpoch=BTNHGV2ParameterClass.stopableEpoch):
		self.patience = patience
		self.min_delta = min_delta
		self.stopableEpoch=stopableEpoch
		self.best_score = None
		self.counter = 0
		self.early_stop = False
		self.best_model_state = None  # 内存保存

	def __call__(self, val_loss, model, epochs):
		if self.best_score is None:
			self.best_score = val_loss
			self.best_model_state = copy.deepcopy(model.state_dict())
		elif val_loss < self.best_score - self.min_delta:
			self.best_score = val_loss
			self.best_model_state = copy.deepcopy(model.state_dict())
			self.counter = 0
		else:
			self.counter += 1
			if self.counter >= self.patience and self.stopableEpoch<=epochs:
				self.early_stop = True
		return self.early_stop

	def restore_best_weights(self, model):
		if self.best_model_state is not None:
			model.load_state_dict(self.best_model_state)
