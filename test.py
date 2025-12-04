



from ModelTrainerTesterClass import ModelTrainerTesterClass
from HANClass import HANClass
from HGTClass import HGTClass
from RGCNClass import RGCNClass

han=HANClass(heteroDataCls=heteroDataClass)
hgt=HGTClass(heteroDataCls=heteroDataClass)
rgcn=RGCNClass(heteroDataCls=heteroDataClass)
han.train()







# from resultAnalysisClass import resultAnalysisClass
# # r=resultAnalysisClass()
# # r._saveBTNHGV2ParameterClass()

# class cClass:
# 	def __init__(self, a):
# 		self.a=a
# 		self.b=self.a*2
# 		self.printc(self.b)
# 	def printc(self, c):
# 		c=self.a*self.b
# 		print(f"c={c}")
# c=cClass(1)
# print(f"a={c.a}")
# print(f"b={c.b}")

# import platform
# import torch
# print(platform.machine())
# print(platform.processor())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

# import cpuinfo

# info = cpuinfo.get_cpu_info()
# print(info['brand_raw'])   # 完整 CPU 型号

