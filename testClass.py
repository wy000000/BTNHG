# class BTNHGV2ParameterClass():
# 	debugMode=False	
# 	_epochs=512

# 	@classmethod
# 	def epochs(cls):
# 		if cls.debugMode:
# 			return int(cls._epochs/10)   # 调试模式缩短为原来的 1/10
# 		return int(cls._epochs)
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
a=BTNHGV2ParameterClass.epochs()
#打印a和类型
print(f"a={a},type(a)={type(a)}")


