from layer import layer
from get_mem import check_gpu_mem
import cupy as cp
import copy

class network:
    def __init__(self, dims, activations, gpus, m):
        assert( len(dims)-1 == len(activations) )
        assert( len(gpus)   == len(activations) )        
        self.m = m
        self.layers = []
        self.gpus = gpus
        for i in range(0,len(dims)-1):
            print("---------- for making layer "+str(i)+" -------------")            
            self.layers.append(layer(dims[i+1],dims[i], gpus[i], activations[i]))
            check_gpu_mem()
        print("####################################################")
        
    def propagate_all(self, X):
        self.A = [X]
        for i in range(0,len(self.layers)):
            print("---------- for propagating layer "+str(i)+" -------------")            
            self.A.append(self.layers[i].propagate(self.A[-1]))
        with cp.cuda.Device(self.gpus[-1]):
            self.output = cp.asarray(self.A[-1])
        return self.A[-1]

    def compute_cost(self, Y):
        # C = 1/(2n) * sum for every example( ||y - A[L]|| ^2 )
        # with cp.cuda.Device(self.gpus[-1]):
            # self.output = cp.asarray(self.output)            
            # self.Y = cp.asarray(Y)
        print("types are Y: {}, and self.output {}".format(str(type(Y)), str(type(self.output))))
        self.cost = 0.5*cp.sum(cp.linalg.norm(Y-self.output,axis=0)**2)        
        return self.cost
                
    def compute_acc(self, Y):
        # with cp.cuda.Device(self.gpus[-1]):
            # self.output_gpu = cp.asarray(self.output)
        self.yhat = (self.output == cp.amax(self.output, axis=0, keepdims=True))
        self.num_correct = 1.*cp.sum(Y*self.yhat)
        return 100.*self.num_correct/self.m
    
    def backprop_all(self, Y):
        # self.error is dJ/dA[L] if cost function J() changes so does this derivative
        dA = self.error = (self.output-Y)
        for i in range(1,len(self.layers)+1):
            dA = self.layers[-i].backprop(dA, self.A[-(i+1)], self.m)

    def update_all(self, alpha=0.001, lambd=0.):
        for l in self.layers:
            l.update(alpha, self.m, lambd)
