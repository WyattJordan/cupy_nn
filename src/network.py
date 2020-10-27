from layer import layer
from get_mem import check_gpu_mem
import cupy as cp
import numpy as np
import copy

class network:
    def __init__(self, dims, activations, m, gpus):
        assert( len(dims)-1 == len(activations) )
        assert( len(gpus)   == len(activations) )        
        self.m = m
        self.layers = []
        self.gpus = gpus
        self.gpu_jump_forward = (np.diff(gpus, prepend=gpus[0])!=0).astype(bool)
        self.gpu_jump_backward = np.append(self.gpu_jump_forward, [0])[1:]
        self.mempool = cp.get_default_memory_pool()
        for i in range(0,len(dims)-1):
            print("---------- making layer "+str(i)+" on gpu "+str(gpus[i])+" -------------")
            self.layers.append(layer(i, dims[i+1], dims[i], int(gpus[i]), activations[i]))
        
    def propagate_all(self, X):
        self.A = [X]
        print("freeing all previous activations... ",end="")
        self.free_mem(0, debug=True)
        for i in range(0,len(self.layers)):
            print("---------- for propagating layer "+str(i)+" -------------")
            self.A.append(self.layers[i].propagate(self.A[-1], self.gpu_jump_forward[i]))
            # self.free_mem(i, debug=True)
            # check = check_gpu_mem(False)
            # print("total is now {:.1f}".format(check[self.gpus[i]][0]))
        self.output = None
        self.free_mem(len(self.layers))
        self.output = self.A[-1]
        return self.output

    def compute_cost(self, Y):
        # C = 1/(2m) * sum for every example( ||y - A[L]|| ^2 )
        with cp.cuda.Device(self.gpus[-1]):
            print("Y dev is: "+str(Y.device)+" output dev is: "+str(self.output.device))
            print("types are Y: {}, and self.output {}".format(str(type(Y)), str(type(self.output))))
            self.cost = 0.5*cp.sum(cp.linalg.norm(Y-self.output,axis=0)**2)
        return self.cost
                
    def compute_acc(self, Y):
        with cp.cuda.Device(self.gpus[-1]):        
            self.yhat = None
            self.free_mem(len(self.layers))
            self.yhat = (self.output == cp.amax(self.output, axis=0, keepdims=True))
            self.num_correct = 1.*cp.sum(Y*self.yhat)
            self.free_mem(-1, debug=True)
            return 100.*self.num_correct/self.m
    
    def backprop_all(self, Y):
        # da (or error) is dJ/dA[L] if cost function J() changes so does this derivative
        with cp.cuda.Device(self.gpus[-1]):                
            dZ = (self.output-Y)
            w_after = cp.array([])       # last layer has no weights in layer after it
        for i in range(1,len(self.layers)+1):
            print("---------- backpropagating layer "+str(i)+" -------------")
            dZ = self.layers[-i].backprop(dZ, w_after, self.A[-(i+1)], self.m,\
                                          self.gpu_jump_forward[-i], self.gpu_jump_backward[-i])
            w_after = self.layers[-i].w
            # self.free_mem(-i, debug=True)

    def update_all(self, alpha=0.001, lambd=0.):
        for l in self.layers:
            l.update(alpha, self.m, lambd)

    # Activations transferred between GPUs only need VRAM temporarily
    def free_mem(self, layer, output=False, debug=False):
        if debug:
            before = check_gpu_mem(False)

        # if not a layer index free memory on all gpus, otherwise just this layer's gpu
        if layer>=len(self.layers) or layer<-1*len(self.layers):
            for g in np.unique(self.gpus):
                with cp.cuda.Device(g):
                    self.mempool.free_all_blocks()
        else:        
            with cp.cuda.Device(self.gpus[layer]):
                self.mempool.free_all_blocks()
            
        if debug:
            diff = before - check_gpu_mem(False)
            if (diff)[:,0].any() != 0:
                if debug:
                    freed = diff[(diff)[:,0]!=0,0]
                    print("{} MiB freed on layer {}".format(freed, layer))
            else:
                print("No memory freed from releasing blocks")
                        
