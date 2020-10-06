import cupy as cp

class sigmoid():
    def fn(self,z):
        return cp.where(z<0, cp.exp(z)/(1.+cp.exp(z)), 1./(1.+cp.exp(-z)))
    
    def dfn(self,z):
        return self.fn(z)*(1. - self.fn(z))

class relu():
    def fn(self, z):
        return cp.clip(z,0.,None)

    def dfn(self, z):
        return cp.where(z<0,0.,1.)

class tanh():
    def fn(self, z):
        return cp.tanh(z)

    def dfn(self, z):
        return 1. - self.fn(z)**2.
    
