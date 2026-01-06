import numpy as np
import torch
from torch.utils.data import TensorDataset

def _build_address_time_feature_dataSet(self, addressDict):
	# 收集所有 clusterID 和特征
	cluster_ids = []
	features_list = []
	for v in addressDict.values():
		cluster_ids.append(v["clusterID"])
		features_list.append(v["addressTimeFeatureCls"].block_features.astype(np.float32))

	# 转为 NumPy 数组
	features_np = np.stack(features_list)   # shape: (N, d1, d2) 或 (N, d)
	cluster_ids = np.array(cluster_ids)

	# 将非连续 clusterID 映射为连续整数
	unique_ids, labels_np = np.unique(cluster_ids, return_inverse=True)

	# 转为 PyTorch Tensor
	features = torch.from_numpy(features_np)
	labels = torch.from_numpy(labels_np).long()

	# 构建 TensorDataset
	dataSet = TensorDataset(features, labels)
	return dataSet

