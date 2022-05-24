# parameter classes
class FixedParameter:
    def __init__(self,value,ptype,index):
        self.value = value
        self.ptype = ptype
        self.index = index
        self.fixed = True
        
class FittingParameter(FixedParameter):
    def __init__(self,value,ptype,index,prior):
        super().__init__(value,ptype,index)
        self.fixed = False
        self.prior = prior