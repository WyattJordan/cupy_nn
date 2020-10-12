import numpy as np
import cupy as cp
import activations
from activations import sigmoid, tanh, relu
from get_mem import check_gpu_mem

class layer:

    def __init__(self, dim, dim_prev, gpu, activation="relu",  init=""):

        mems_before = check_gpu_mem(False)

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


        print("initializing layer with dim {} and dim_prev {}".format(dim, dim_prev))
        np.random.seed(1)   # consistent randomization = debuggable network
        self.w = np.random.randn(dim, dim_prev)*factor
        self.b = np.zeros([dim, 1])
        self.gpu = gpu        
        with cp.cuda.Device(self.gpu): 
            self.w = cp.asarray(self.w, dtype=cp.float32)
            self.b = cp.asarray(self.b, dtype=cp.float32)
        # print("size of w is "+str(self.w.nbytes/2**20)+" and b is "+str(self.b.nbytes/2**20))
        mems_after = check_gpu_mem(False)
        # print("memory added is {} MiB and total is {} on GPU1"\
              # .format(mems_after[1][0] - mems_before[1][0], mems_after[1][0]))
        
    def propagate(self, A_prev):
        with cp.cuda.Device(self.gpu):
            #print("self.w shape "+str(self.w.shape)+" A_prev.shape "+str(A_prev.shape))
            mems_before = check_gpu_mem(False)
            # if crossing over to this gpu:
            # A_prev = cp.asarray(A_prev, dtype=cp.float32)
            self.Z = cp.dot(self.w, A_prev)+self.b
            self.A = self.activation.fn(self.Z)
            mems_after = check_gpu_mem(False)
            exp_added = (self.Z.nbytes + self.A.nbytes)/2**20
            print("size of Z and A is "+str(self.Z.nbytes/2**20)+" for total "+str(exp_added))
            print("size of A_prev is "+str(A_prev.nbytes/2**20))
            
            print("memory added is {} MiB and total is {} on GPU1"\
                  .format(mems_after[1][0] - mems_before[1][0], mems_after[1][0]))
            # add dropout here and scale self.A as needed (scaling also required in backprop!)
            print("error for mem added: "+str(mems_after[1][0] - mems_before[1][0] - exp_added))
            return self.A

    # next layer already calculated dA for this layer
    # find dA, dw, db for this layer and dA for previous layer
    def backprop(self, dA, A_prev, m):
        with cp.cuda.Device(self.gpu):
            A_prev = cp.asarray(A_prev, dtype=cp.float32)
            dA = cp.asarray(dA, dtype=cp.float32)
            self.dZ = dA * self.activation.dfn(self.Z)
            self.dA_prev = cp.dot(self.w.T, self.dZ)
            self.dw = 1/m*cp.dot(self.dZ, A_prev.T)
            self.db = 1/m*cp.sum(self.dZ, axis=1, keepdims=True)
        return self.dA_prev
        
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
