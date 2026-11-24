import os
import time

import torch
from torch.utils.data import TensorDataset

from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass


class NEW2_addressTimeDataCompressNoIntervalClass(addressTimeDataClass):
	dataset_file_name = "NEW2_addressTimeFeature_compress_no_interval_dataSet.pt"

	def __init__(
		self,
		dataPath=BTNHGV2ParameterClass.dataPath,
		log_addressTimeFeature_amount=BTNHGV2ParameterClass.log_addressTimeFeature_amount,
		try_read_save_addressTimeFeature_dataSet=True,
	):
		super().__init__(
			dataPath=dataPath,
			compress=True,
			compress_padding=False,
			noisy_0=False,
			log_addressTimeFeature_amount=log_addressTimeFeature_amount,
			try_read_save_addressTimeFeature_dataSet=try_read_save_addressTimeFeature_dataSet,
		)

	def compress_address_time_feature_dataSet(self):
		dataSet = self.addressTimeFeature_dataSet
		if dataSet is None:
			raise ValueError("addressTimeFeature_dataSet is None")

		features, labels = dataSet.tensors
		num_samples = features.shape[0]
		print(
			f"NEW2 compress no interval before: num_samples={num_samples}, "
			f"features.shape={features.shape}, labels.shape={labels.shape}"
		)

		time1 = time.time()
		total_before = features.shape[0] * features.shape[1]
		filtered_samples = []

		for i in range(num_samples):
			sample_features = features[i]
			filtered_sample = sample_features[sample_features[:, 0] != -1]
			filtered_samples.append(filtered_sample)

			if (i + 1) % 2001 == 0:
				print("finished {} rows, progress {:.2f}%".format(i, i / num_samples * 100))

		max_len = max(sample.shape[0] for sample in filtered_samples)
		feature_dim = filtered_samples[0].shape[1]
		padded_features = torch.zeros((num_samples, max_len, feature_dim), device=features.device)

		for i, sample in enumerate(filtered_samples):
			padded_features[i, :sample.shape[0], :] = sample

		padded_dataset = TensorDataset(padded_features, labels)
		time2 = time.time()
		usedTime = time2 - time1

		print(f"NEW2 compress no interval after: padded_features.shape={padded_features.shape}, labels.shape={labels.shape}")
		total_after = padded_features.shape[0] * padded_features.shape[1]
		print(f"NEW2 compression reduction: {1 - (total_after / total_before):.2%}")
		print(
			"NEW2 compress no interval used time: {}h {}m {}s".format(
				int(usedTime // 3600), int(usedTime // 60), int(usedTime % 60)
			)
		)

		self.addressTimeFeature_dataSet = padded_dataset
		return padded_dataset

	def _dataset_path(self, path=None):
		if path is None:
			path = self._dataPath
		return os.path.join(path, self.dataset_file_name)

	def save_tensor_dataset(self):
		dataset = self.addressTimeFeature_dataSet
		if dataset is None:
			raise ValueError("addressTimeFeature_dataSet is None, please build it first.")

		fullPath = self._dataset_path()
		print(f"saving NEW2 compress no interval addressTimeFeature_dataSet to {fullPath}")
		torch.save({f"tensor_{i}": t for i, t in enumerate(dataset.tensors)}, fullPath)
		print("saved")
		return fullPath

	def load_tensor_dataset(self, path=None):
		try:
			fullPath = self._dataset_path(path)
			print(f"loading NEW2 compress no interval addressTimeFeature_dataSet from {fullPath}")
			data_dict = torch.load(fullPath)
			tensors = [data_dict[key] for key in sorted(data_dict.keys())]
			self.addressTimeFeature_dataSet = TensorDataset(*tensors)
			print("loaded")
			return True
		except Exception as e:
			print(f"load NEW2 compress no interval tensorDataset failed: {e}")
			return False
