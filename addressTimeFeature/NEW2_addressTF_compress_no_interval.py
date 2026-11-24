"""表 1 第 2 行：保留活动顺序、丢弃非活跃区块，但不加区间通道（13 维）。

与主实验（压缩 + 区间，14 维）唯一的差别就是没有显式的“距上一次活动”通道。

运行方式（在仓库根目录下）：
	python addressTimeFeature/NEW2_addressTF_compress_no_interval.py
"""
from set_parent_dir import setup_parent_dir
setup_parent_dir()

print("start import NEW2 compress no interval experiment")
import time

time1 = time.time()

from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from DataSetModelTrainerTesterClass import DataSetModelTrainerTesterClass
from NEW2_addressTimeDataCompressNoIntervalClass import NEW2_addressTimeDataCompressNoIntervalClass
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
	print(f"start NEW2 compress no interval run {run_i + 1}/{run_times}")

	model = build_model(addressTimeDataCls)
	trainerTesterCls = DataSetModelTrainerTesterClass(
		model=model,
		addressTimeDataCls=addressTimeDataCls,
		resultFolderName="NEW2_compress_no_interval",
	)

	resultAnalyCls = trainerTesterCls.kFold_train_test()
	resultAnalyCls.save_kFold()

	print(f"finish NEW2 compress no interval run {run_i + 1}/{run_times}")


if __name__ == "__main__":
	run_times = BTNHGV2ParameterClass.addressTF_run_times
	print(f"NEW2 compress no interval: {run_times} run(s) x {BTNHGV2ParameterClass.kFold_k}-fold")
	addressTimeDataCls = NEW2_addressTimeDataCompressNoIntervalClass()

	for run_i in range(run_times):
		run_once(run_i, run_times, addressTimeDataCls)
