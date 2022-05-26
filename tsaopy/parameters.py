# parameter classes
class Fix:
    def __init__(self,value,ptype,index):
        self.value = value
        self.ptype = ptype
        self.index = index
        self.fixed = True
        
class Fit(Fix):
    def __init__(self,value,ptype,index,prior):
        super().__init__(value,ptype,index)
        self.fixed = False
        self.prior = prior
        
