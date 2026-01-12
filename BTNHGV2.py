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
# from GraphConvClass import GraphConvClass

time2 = time.time()
print("import used time: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

# 处理数据集
heteroDataCls=BTNHGV2HeteroDataClass()

# 定义模型
# gmodel=HANClass(heteroData=heteroDataCls.heteroData)
# gmodel=HGTClass(heteroData=heteroDataCls.heteroData)
# gmodel=RGCNClass(heteroData=heteroDataCls.heteroData)
gmodel=SAGEClass(heteroData=heteroDataCls.heteroData)
# gmodel=GATClass(heteroData=heteroDataCls.heteroData)
# gmodel=GraphConvClass(heteroData=heteroDataCls.heteroData)

# 准备训练器测试器
trainertester=HeteroModelTrainerTesterClass(model=gmodel)

#单次训练测试
resultAnalyCls=trainertester.train_test()
resultAnalyCls.save()

#辅助显示
resultAnalyCls.showEvaluationMetrics()
resultAnalyCls.showExtendedAttributes()
resultAnalyCls.plot_true_pred_counts()
resultAnalyCls.plot_confusion_matrix()

# 交叉验证
resultAnalyCls=trainertester.kFold_train_test()
resultAnalyCls.save_kFold()
