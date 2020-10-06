from layer import layer
import numpy as np
import copy

class network:
    def __init__(self, dims, activations, m):
        self.m = m
        self.layers = []
        for i in range(1,len(dims)):
            self.layers.append(layer(dims[i],dims[i-1], i, activations[i-1]))

    def propagate_all(self, X):
        self.A = [X]
        for i in range(0,len(self.layers)):
            self.A.append(self.layers[i].propagate(self.A[-1]))
        return self.A[-1]

    def compute_cost(self, output, Y):
        # C = 1/(2n) * sum for every example( ||y - A[L]|| ^2 )
        self.cost = 0.5*np.sum(np.linalg.norm(Y-output,axis=0)**2)        
        return self.cost
                
    def compute_acc(self, output, Y):
        self.yhat = (output == np.amax(output, axis=0, keepdims=True))
        self.num_correct = 1.*np.sum(Y*self.yhat)
        return 100.*self.num_correct/self.m
    
    def backprop_all(self, output, Y):
        # self.error is dJ/dA[L] if cost function J() changes so does this derivative
        dA = self.error = (output-Y)
        for i in range(1,len(self.layers)+1):
            dA = self.layers[-i].backprop(dA, self.A[-(i+1)], self.m)

    def update_all(self, alpha=0.001, lambd=0.):
        for l in self.layers:
            l.update(alpha, self.m, lambd)
