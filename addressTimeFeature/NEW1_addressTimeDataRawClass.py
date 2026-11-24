import os

import torch
from torch.utils.data import TensorDataset

from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass


class NEW1_addressTimeDataRawClass(addressTimeDataClass):
	dataset_file_name = "NEW1_addressTimeFeature_raw_dataSet.pt"

	def __init__(
		self,
		dataPath=BTNHGV2ParameterClass.dataPath,
		log_addressTimeFeature_amount=BTNHGV2ParameterClass.log_addressTimeFeature_amount,
		try_read_save_addressTimeFeature_dataSet=False,
	):
		super().__init__(
			dataPath=dataPath,
			compress=False,
			compress_padding=False,
			noisy_0=False,
			log_addressTimeFeature_amount=log_addressTimeFeature_amount,
			try_read_save_addressTimeFeature_dataSet=try_read_save_addressTimeFeature_dataSet,
		)

	def _dataset_path(self, path=None):
		if path is None:
			path = self._dataPath
		return os.path.join(path, self.dataset_file_name)

	def save_tensor_dataset(self):
		dataset = self.addressTimeFeature_dataSet
		if dataset is None:
			raise ValueError("addressTimeFeature_dataSet is None, please build it first.")

		fullPath = self._dataset_path()
		print(f"saving NEW1 raw addressTimeFeature_dataSet to {fullPath}")
		torch.save({f"tensor_{i}": t for i, t in enumerate(dataset.tensors)}, fullPath)
		print("saved")
		return fullPath

	def load_tensor_dataset(self, path=None):
		try:
			fullPath = self._dataset_path(path)
			print(f"loading NEW1 raw addressTimeFeature_dataSet from {fullPath}")
			data_dict = torch.load(fullPath)
			tensors = [data_dict[key] for key in sorted(data_dict.keys())]
			self.addressTimeFeature_dataSet = TensorDataset(*tensors)
			print("loaded")
			return True
		except Exception as e:
			print(f"load NEW1 raw tensorDataset failed: {e}")
			return False
