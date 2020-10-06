import numpy as np

class sigmoid():
    def fn(self,z):
        return np.where(z<0, np.exp(z)/(1.+np.exp(z)), 1./(1.+np.exp(-z)))
    
    def dfn(self,z):
        return self.fn(z)*(1. - self.fn(z))

class relu():
    def fn(self, z):
        return np.clip(z,0.,None)

    def dfn(self, z):
        return np.where(z<0,0.,1.)

class tanh():
    def fn(self, z):
        return np.tanh(z)

    def dfn(self, z):
        return 1. - self.fn(z)**2.
    
