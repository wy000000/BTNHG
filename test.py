class c():
    def __init__(self):
        self.a=1
    def seta(self):
        self.seta5()
    def seta5(self,v=5):
        self.a=v
c=c()
print(c.a)
c.seta()
print(c.a)
