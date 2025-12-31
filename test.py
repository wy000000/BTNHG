class c:
	a=1
	def __init__(self):
		self.b=2
	def f(self):
		print("c.a=", c.a)
c1=c()
c1.f()