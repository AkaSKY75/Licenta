class Parinte:
    def __init__(self, prop1, prop2):
        self.prop1 = prop1
        self.prop2 = prop2
    def PropShow(self):
        print(self.prop1+" "+self.prop2)

class Copil(Parinte):
    def __init__(self, prop1, prop2, prop3, prop4):
        super(Copil, self).__init__(prop1, prop2)
        self.prop3 = prop3
        self.prop4 = prop4
    def PropShow(self):
        print(self.prop1+" "+self.prop2+" "+self.prop3+" "+self.prop4)

p = Parinte('a', 'b')
c = Copil('c', 'd', 'e', 'f')

p.PropShow()
c.PropShow()
