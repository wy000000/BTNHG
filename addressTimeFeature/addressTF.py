import set_parent_dir
print("start import")
import time
time1 = time.time()
import sys
import os
# 获取当前文件所在目录的父目录
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# sys.path.append(parent_dir)
import torch
import torch.nn as nn
import datetime
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import importlib
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeFeature.addressTimeFeatureClass import addressTimeFeatureClass
from EarlyStoppingClass import EarlyStoppingClass
from resultAnalysisClass import resultAnalysisClass
from ExtendedNNModule import ExtendedNNModule
from addressTimeDataClass import addressTimeDataClass
from addressTimeFeature.CNN1D_DW_class import CNN1D_DW_class
from DataSetModelTrainerTesterClass import DataSetModelTrainerTesterClass
from simple2DCNNClass import simple2DCNNClass
from CNN1D_DW_class import CNN1D_DW_class
from CNN1D_DW_PW_class import CNN1D_DW_PW_class
from CNN1D_DW_SE_class import CNN1D_DW_SE_class
from CNN1D_DW_SE_TF_class import CNN1D_DW_SE_TF_class
from CNN1D_DW_TF_class import CNN1D_DW_TF_class
from CNN1D_TF_class import CNN1D_TF_class
from CNN1D_DW_SE_PE_TF_class import CNN1D_DW_SE_PE_TF_class

time2 = time.time()
print("import used time: ", time2 - time1)

# 初始化数据类
addressTimeDataCls=addressTimeDataClass()

# 初始化模型类
# model=CNN1D_DW_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
# model=CNN1D_DW_PW_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
# model=CNN1D_DW_SE_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
# model=CNN1D_DW_SE_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
# model=CNN1D_DW_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
# model=CNN1D_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)
model=CNN1D_DW_SE_PE_TF_class(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)

# 初始化训练器测试器类
TrainerTesterCls=DataSetModelTrainerTesterClass(model=model, addressTimeDataCls=addressTimeDataCls)

# #单次
# resultAnalyCls=TrainerTesterCls.train_test()
# resultAnalyCls.save()

#kfold
resultAnalyCls=TrainerTesterCls.kFold_train_test()
resultAnalyCls.save_kFold()

# #显示单次
# resultAnalyCls.showEvaluationMetrics()
# resultAnalyCls.showExtendedAttributes()
# resultAnalyCls.plot_true_pred_counts()
# resultAnalyCls.plot_confusion_matrix()
a=1