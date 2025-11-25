print("start import")
import time
time1 = time.time()
import torch
from torch_geometric.data import Data # 从torch_geometric.data导入Data类，用于表示图数据
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn

time2 = time.time()
print("import used time: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
heteroDataClass=BTNHGV2HeteroDataClass()

from ModelTrainerClass import ModelTrainerClass
from HAN import HAN
gmodel=HAN(heteroDataCls=heteroDataClass,
			hidden_channels=BTNHGV2ParameterClass.hidden_channels,
			out_channels=BTNHGV2ParameterClass.out_channels,
			num_heads=BTNHGV2ParameterClass.num_heads,
			dropout=BTNHGV2ParameterClass.dropout
			)

trainer=ModelTrainerClass(model=gmodel,
						  	device=None,
							lr=BTNHGV2ParameterClass.lr,
							weight_decay=BTNHGV2ParameterClass.weight_decay,
							epochs=BTNHGV2ParameterClass.epochs,
							patience=BTNHGV2ParameterClass.patience,
							loss_threshold=BTNHGV2ParameterClass.loss_threshold
						)
trainer.run()
trainer.test()