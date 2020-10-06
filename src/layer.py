import activations
import numpy as np
from activations import sigmoid, tanh, relu
class layer:

    def __init__(self, dim, dim_prev, seed=1, activation="relu", init=""):

        if activation=="sigmoid":
            self.activation = sigmoid()
            factor = 1. / np.sqrt(dim_prev)
        elif activation=="tanh":
            self.activation = tanh()
            factor = 1. / np.sqrt(dim_prev)
        else: # assumes activation="relu":
            self.activation = relu();
            factor = 2. / np.sqrt(dim_prev)

        if init=="Xavier":
            factor = np.sqrt(6. / (dim_prev + dim))
        elif init.replace(".","",1).isnumeric():
            factor = float(init_factor)

        np.random.seed(seed)   # consistent randomization = debuggable network
        self.w = np.random.randn(dim, dim_prev)*factor
        self.b = np.zeros([dim, 1])

    def propagate(self, A_prev):
        self.Z = np.dot(self.w, A_prev)+self.b
        self.A = self.activation.fn(self.Z)
        # add dropout here and scale self.A as needed (scaling also required in backprop!)
        return self.A

    # next layer already calculated dA for this layer
    # find dA, dw, db for this layer and dA for previous layer
    def backprop(self, dA, A_prev, m):
        self.dA = dA
        self.dZ = dA * self.activation.dfn(self.Z)
        self.dA_prev = np.dot(self.w.T, self.dZ)
        self.dw = 1/m*np.dot(self.dZ, A_prev.T)
        self.db = 1/m*np.sum(self.dZ, axis=1, keepdims=True)
        return self.dA_prev
        
    # assumes L2 regularization or no regularization
    def update(self, alpha=0.001, m=1., lambd=0.):
        self.w = self.w - (alpha * self.dw) #(self.dw + lambd/m*self.w))
        self.b = self.b - (alpha * self.db)

    # Equality Check for layers when net.check_gradient exits
    def __eq__(self,other):
        if not isinstance(other, layer):
            return NotImplemented
        
        attributes_to_check = ["w","b", "dw", "db"]
        equal = True
        for a in attributes_to_check:
            if not getattr(self,a).all() == getattr(other,a).all():
                print("layers have different attribute "+str(a))
                equal = False
                break
        return equal
