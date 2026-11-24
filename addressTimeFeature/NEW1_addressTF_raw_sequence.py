"""表 1 第 1 行：完整原始序列（不压缩，5,621 个时间步）。

该配置在论文所用硬件上会耗尽 GPU 显存（OOM），保留此脚本是为了让
“完整序列不可训练”这一结论可复现，而不是为了让该行得到一个准确率。

运行方式（在仓库根目录下）：
	python addressTimeFeature/NEW1_addressTF_raw_sequence.py
"""
from set_parent_dir import setup_parent_dir
setup_parent_dir()

print("start import NEW1 raw sequence experiment")
import time

time1 = time.time()

from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from DataSetModelTrainerTesterClass import DataSetModelTrainerTesterClass
from NEW1_addressTimeDataRawClass import NEW1_addressTimeDataRawClass
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
	# model=CNN1D_DW_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	# model=CNN1D_DW_PW_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	# model=CNN1D_DW_SE_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	# model=CNN1D_DW_SE_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	model=CNN1D_DW_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	# model=CNN1D_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	# model=CNN1D_DW_SE_PE_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	# model=CNN1D_DW_SE_PE_TF_CLS_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	# model=simple2DCNNClass(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
	return model


def run_once(run_i, run_times, addressTimeDataCls):
	print(f"start NEW1 raw sequence run {run_i + 1}/{run_times}")

	model = build_model(addressTimeDataCls)
	trainerTesterCls = DataSetModelTrainerTesterClass(
		model=model,
		addressTimeDataCls=addressTimeDataCls,
		resultFolderName="NEW1_raw_sequence",
	)

	resultAnalyCls = trainerTesterCls.kFold_train_test()
	resultAnalyCls.save_kFold()

	print(f"finish NEW1 raw sequence run {run_i + 1}/{run_times}")


if __name__ == "__main__":
	run_times = BTNHGV2ParameterClass.addressTF_run_times
	print(f"NEW1 raw sequence: {run_times} run(s) x {BTNHGV2ParameterClass.kFold_k}-fold")
	addressTimeDataCls = NEW1_addressTimeDataRawClass()

	for run_i in range(run_times):
		run_once(run_i, run_times, addressTimeDataCls)
