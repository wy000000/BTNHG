import set_parent_dir
print("start import")
import time
time1 = time.time()
import sys
import os
# 获取当前文件所在目录的父目录
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# sys.path.append(parent_dir)
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from addressTimeFeature.addressTimeFeatureClass import addressTimeFeatureClass
from EarlyStoppingClass import EarlyStoppingClass
from resultAnalysisClass import resultAnalysisClass
from ExtendedNNModule import ExtendedNNModule
from addressTimeDataClass import addressTimeDataClass
from simple1DCNNClass import simple1DCNNClass
from DataSetModelTrainerTesterClass import DataSetModelTrainerTesterClass
import torch
import torch.nn as nn
import datetime
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import importlib
time2 = time.time()
print("import used time: ", time2 - time1)

# 初始化数据类
addressTimeDataCls=addressTimeDataClass()

# 初始化模型类
model=simple1DCNNClass(addressTimeFeature_dataSet=addressTimeDataCls.addressTimeFeature_dataSet)

# 初始化训练器测试器类
TrainerTesterCls=DataSetModelTrainerTesterClass(model=model, addressTimeDataCls=addressTimeDataCls)
resultAnalyCls=TrainerTesterCls.train_test()
# resultAnalyCls=TrainerTesterCls.kFold_train_test()

# 保存结果分析类
resultAnalyCls.save()
# resultAnalyCls.save_kFold()

# resultAnalyCls.showEvaluationMetrics()
# resultAnalyCls.showExtendedAttributes()
# resultAnalyCls.plot_true_pred_counts()
# resultAnalyCls.plot_confusion_matrix()