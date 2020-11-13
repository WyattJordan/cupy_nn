import numpy as np
import cupy as cp
import activations
from activations import sigmoid, tanh, relu
from get_mem import check_gpu_mem

class layer:

    def __init__(self, layer_num, dim, dim_prev, gpu, activation="relu", precision=cp.float32, init=""):
        self.lay_num = layer_num
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


        np.random.seed(1)   # consistent randomization = debuggable network
        self.w = np.random.randn(dim, dim_prev)*factor
        self.b = np.zeros([dim, 1], dtype=np.float16)
        self.gpu = gpu        
        with cp.cuda.Device(self.gpu): 
            self.w = cp.asarray(self.w, dtype=precision)
            self.b = cp.asarray(self.b, dtype=precision)
        
    def propagate(self, A_prev):
        with cp.cuda.Device(self.gpu):
            # free the memory from the previous Z and A to be replaced by their new values
            self.Z = self.A = None
            # print('A_prev shape is: {}, self.w shape is {}'.format(A_prev.shape, self.w.shape))
            self.Z = cp.dot(self.w, A_prev) + self.b
            self.A = self.activation.fn(self.Z)
            # # add dropout here and scale self.A as needed (scaling also required in backprop!)
            return self.A

    def backprop(self, dZ_after, w_after, A_before, m):
        with cp.cuda.Device(self.gpu):
            # mark variables as free memory before re-assigning (doesn't double memory usage)
            self.dZ = self.dw = self.db = None
            
            if w_after.size == 0 : # last layer in the network, dZ given based on error
                self.dZ = dZ_after
                self.dw = 1/m*cp.dot(self.dZ, A_before.T)
                self.db = 1/m*cp.sum(self.dZ, axis=1, keepdims=True)
                return self.dZ

            self.dZ = cp.dot(w_after.T, dZ_after) * self.activation.dfn(self.Z)
            self.dw = 1/m*cp.dot(self.dZ, A_before.T)
            self.db = 1/m*cp.sum(self.dZ, axis=1, keepdims=True)
        return self.dZ
        
    # assumes L2 regularization or no regularization
    def update(self, alpha=0.001, m=1., lambd=0.):
        with cp.cuda.Device(self.gpu):
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
