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
time2 = time.time()
print("import used time: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")


heteroDataClass=BTNHGV2HeteroDataClass()

from ModelTrainerClass import ModelTrainerClass
from HANClass import HANClass
from HGTClass import HGTClass
from RGCNClass import RGCNClass

gmodel=HANClass(heteroDataCls=heteroDataClass)
# gmodel=HGTClass(heteroDataCls=heteroDataClass)
# gmodel=RGCNClass(heteroDataCls=heteroDataClass)

trainer=ModelTrainerClass(model=gmodel)

trainer.train()

trainer.test()