from layer import layer
from get_mem import check_gpu_mem
import cupy as cp
import numpy as np
import copy

class network:
    def __init__(self, dims, activations, m, gpus, precisions, lambd):
        self.mempool = cp.get_default_memory_pool()    
        assert( len(dims)-1 == len(activations) )
        assert( len(gpus)   == len(activations) )        
        self.m = m
        self.lambd = lambd
        self.layers = []
        self.gpus = gpus
        for i in range(0,len(dims)-1):
            print("---------- making layer "+str(i)+" on gpu "+str(gpus[i])+" -------------")
            self.layers.append(layer(i, dims[i+1], dims[i], int(gpus[i]), activations[i], precisions[i]))
        
    def propagate_all(self, X):
        self.A = [X]
        for i in range(0,len(self.layers)):
            self.A.append(self.layers[i].propagate(self.A[-1]))
        self.output = None
        self.output = self.A[-1]
        return self.output

    def compute_cost(self, Y):
        # C = 1/(2m) * sum for every example( ||y - A[L]|| ^2 )
        with cp.cuda.Device(self.gpus[-1]):
            # print("Y dev is: "+str(Y.device)+" output dev is: "+str(self.output.device))
            # print("types are Y: {}, and self.output {}".format(str(type(Y)), str(type(self.output))))
            L2 = 0.            
            if self.lambd != 0:
                for l in self.layers:
                    L2 += self.lambd/(2*self.m) * cp.sum(cp.dot(l.w, l.w))
                    
            self.cost = float(cp.sum(cp.linalg.norm(Y-self.output,axis=0)**2) + L2)
        return self.cost
                
    def compute_acc(self, Y):
        with cp.cuda.Device(self.gpus[-1]):        
            self.yhat = None
            # self.free_mem(len(self.layers), debug=True)
            self.yhat = (self.output == cp.amax(self.output, axis=0, keepdims=True))
            self.num_correct = 1.*cp.sum(Y*self.yhat)
#            self.free_mem(-1, debug=True)
            return float(100.*self.num_correct/self.m)
    
    def backprop_all(self, Y):
        # da (or error) is dJ/dA[L] if cost function J() changes so does this derivative
        with cp.cuda.Device(self.gpus[-1]):                
            dZ = (self.output-Y)
            w_after = cp.array([])       # last layer has no weights in layer after it
        for i in range(1,len(self.layers)+1):
            dZ = self.layers[-i].backprop(dZ, w_after, self.A[-(i+1)], self.m)
            w_after = self.layers[-i].w
            # self.free_mem(-i, debug=True)

    def update_all(self, alpha=0.001):
        for l in self.layers:
            l.update(alpha, self.m, self.lambd)

    def evaulate(self, dl, dset):
        print("Starting evaluation on "+dset)
        accs = np.array([])
        costs = np.array([])            
        while 1:
            x, y, loop, sz = dl.get_next_batch(dset, gpu)
            output = self.propagate_all(x)
            costs = np.append(costs, self.compute_cost(y))
            accs = np.append(accs, self.compute_acc(y))

            if loop:
                batch_szs = np.append(np.ones(len(accs)-1)*batchsz,sz)
                print(batch_szs)
                weights = batch_szs / np.sum(batch_szs)
                acc = np.sum(accs * weights)
                cost = np.sum(costs * weights) 
                break
        return cost, acc
        
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
                        
