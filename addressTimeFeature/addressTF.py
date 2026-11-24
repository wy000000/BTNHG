"""CNN1D_DW_TF 主实验（论文表 2 的最终模型）。

运行方式（在仓库根目录下）：
	python addressTimeFeature/addressTF.py

论文协议为 ``BTNHGV2ParameterClass.addressTF_run_times`` 轮 x 5 折；
每轮会重抽折划分与参数初始化，结果写入
``<dataPath>/result/CNN1D_DW_TF_class-kFold-<时间>-acc <均值>/``。
"""
from set_parent_dir import setup_parent_dir
setup_parent_dir()

print("start import")
import time

time1 = time.time()

from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeDataClass import addressTimeDataClass
from DataSetModelTrainerTesterClass import DataSetModelTrainerTesterClass
from CNN2D import simple2DCNNClass
from CNN1D_DW_class import CNN1D_DW_class
from CNN1D_DW_PW_class import CNN1D_DW_PW_class
from CNN1D_DW_SE_class import CNN1D_DW_SE_class
from CNN1D_DW_SE_TF_class import CNN1D_DW_SE_TF_class
from CNN1D_DW_TF_class import CNN1D_DW_TF_class
from CNN1D_TF_class import CNN1D_TF_class
from CNN1D_DW_SE_PE_TF_class import CNN1D_DW_SE_PE_TF_class
from CNN1D_DW_SE_PE_TF_CLS_class import CNN1D_DW_SE_PE_TF_CLS_class


time2 = time.time()
print("import used time: ", time2 - time1)


def build_model(addressTimeDataCls):
	"""选择要训练的结构；默认即论文的最终模型 CNN1D_DW_TF（表 2 最后一行）。"""
	# ---- 表 1：压缩方式消融由 NEW1/NEW2 脚本负责，这里固定为“压缩+区间”数据集 ----

	# ---- 表 2：结构消融，取消注释其中一个 ----
	# model=simple2DCNNClass(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)   # CNN2D
	# model=CNN1D_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)     # TF only
	# model=CNN1D_DW_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)    # DW
	# model=CNN1D_DW_SE_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet) # DW+SE
	# model=CNN1D_DW_PW_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet) # DW+PW
	model=CNN1D_DW_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)   # DW+TF（最终模型）
	# model=CNN1D_DW_SE_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	# model=CNN1D_DW_SE_PE_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	# model=CNN1D_DW_SE_PE_TF_CLS_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)

	return model


def run_once(run_i, run_times, addressTimeDataCls):
	print(f"start addressTF run {run_i + 1}/{run_times}")

	model = build_model(addressTimeDataCls)
	trainerTesterCls = DataSetModelTrainerTesterClass(
		model=model,
		addressTimeDataCls=addressTimeDataCls
	)

	resultAnalyCls = trainerTesterCls.kFold_train_test()
	resultAnalyCls.save_kFold()

	print(f"finish addressTF run {run_i + 1}/{run_times}")


if __name__ == "__main__":

	run_times = BTNHGV2ParameterClass.addressTF_run_times
	print(f"addressTF: {run_times} run(s) x {BTNHGV2ParameterClass.kFold_k}-fold")
	addressTimeDataCls = addressTimeDataClass()

	for run_i in range(run_times):
		run_once(run_i, run_times, addressTimeDataCls)
