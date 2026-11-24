"""图模型驱动：论文表 3 的八个图基线。

运行方式（在仓库根目录下）：
	python BTNHGV2.py

论文协议为 ``BTNHGV2ParameterClass.graph_run_times`` 轮 x 5 折；每轮先重抽
一次折划分（``getTrainTestMaskKFold``），再重建模型。结果写入
``<dataPath>/result/<模型类名>-kFold-<时间>-acc <均值>/``。
"""
print("start import")
import time
time1 = time.time()
import torch
from torch_geometric.data import Data # 从torch_geometric.data导入Data类，用于表示图数据
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
from HeteroModelTrainerTesterClass import HeteroModelTrainerTesterClass
from HANClass import HANClass
from HGTClass import HGTClass
from RGCNClass import RGCNClass
from resultAnalysisClass import resultAnalysisClass
from SAGEClass import SAGEClass
from GATClass import GATClass
from GraphConvClass import GraphConvClass
from SAGE_TF import SAGE_TF
from GINE_TF import GINE_TF, GPSDataEnhancer

time2 = time.time()
print("import used time: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")


# GINETF 需要一次性为异构图补上边特征（按 6M 条边逐条计算，较慢），
# 多轮实验复用同一份增强后的图，避免每轮重算。
_GINE_HETERO = {}


def _gine_tf_model(heteroDataCls):
	if "hetero" not in _GINE_HETERO:
		_GINE_HETERO["hetero"] = GPSDataEnhancer.enhance_with_edge_features(heteroDataCls)
	return GINE_TF(heteroData=_GINE_HETERO["hetero"])


def build_model(heteroDataCls):
	"""选择要训练的图模型；默认 SAGE，与旧版脚本行为一致。"""
	# ---- 论文表 3 的八个基线，取消注释其中一个 ----
	# model = GraphConvClass(heteroData=heteroDataCls.heteroData)   # GCN（PyG GraphConv）
	# model = GATClass(heteroData=heteroDataCls.heteroData)         # GAT
	model = SAGEClass(heteroData=heteroDataCls.heteroData)          # SAGE
	# model = HANClass(heteroData=heteroDataCls.heteroData)         # HAN
	# model = HGTClass(heteroData=heteroDataCls.heteroData)         # HGT
	# model = RGCNClass(heteroData=heteroDataCls.heteroData)        # RGCN
	# model = SAGE_TF(heteroData=heteroDataCls.heteroData)          # SAGETF
	# model = _gine_tf_model(heteroDataCls)                         # GINETF
	return model


def run_once(run_i, run_times, heteroDataCls):
	print(f"start BTNHG run {run_i + 1}/{run_times}")

	# 每轮重抽折划分：与论文“ten rounds x five folds”协议一致
	heteroDataCls.getTrainTestMaskKFold()

	gmodel = build_model(heteroDataCls)
	trainertester = HeteroModelTrainerTesterClass(model=gmodel)

	resultAnalyCls = trainertester.kFold_train_test()
	resultAnalyCls.save_kFold()

	print(f"finish BTNHG run {run_i + 1}/{run_times}")


if __name__ == "__main__":

	run_times = BTNHGV2ParameterClass.graph_run_times
	print(f"BTNHG: {run_times} run(s) x {BTNHGV2ParameterClass.kFold_k}-fold")

	# 处理数据集
	heteroDataCls = BTNHGV2HeteroDataClass()

	for run_i in range(run_times):
		run_once(run_i, run_times, heteroDataCls)
