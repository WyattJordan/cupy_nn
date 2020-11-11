import numpy as np
import cupy as cp
import activations
from activations import sigmoid, tanh, relu
from get_mem import check_gpu_mem

class layer:

    def __init__(self, layer_num, dim, dim_prev, gpu, activation="relu",  init=""):
        self.mempool = cp.get_default_memory_pool()
        mems_before = check_gpu_mem(False)
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
        self.b = np.zeros([dim, 1])
        self.gpu = gpu        
        with cp.cuda.Device(self.gpu): 
            self.w = cp.asarray(self.w, dtype=cp.float32)
            self.b = cp.asarray(self.b, dtype=cp.float32)
        mems_after = check_gpu_mem(False)
        mem_added = mems_after[gpu][0] - mems_before[gpu][0]
        total = self.w.nbytes/2**20 + self.b.nbytes/2**20
        print("dim {} dim_prev {} w: {:.3f} b: {:.3f} total {:.3f} mem added: {:.1f} error: {:.3f}"\
              .format(dim, dim_prev, self.w.nbytes/2**20, self.b.nbytes/2**20,\
               total, mem_added, mem_added-total))
        
    def propagate(self, A_prev, jump):
        with cp.cuda.Device(self.gpu):
            print("self.w shape "+str(self.w.shape)+" A_prev.shape "+str(A_prev.shape))
            mems_before = check_gpu_mem(False)
            
            if jump:
                print("moving data to correct gpu")
                A_prev = cp.copy(A_prev)#, dtype=cp.float32)
                print("w type: "+str(self.w.dtype)+" shape "+str(self.w.shape)+" dev "+str(self.w.device))
                print("b type: "+str(self.b.dtype)+" shape "+str(self.b.shape)+" dev "+str(self.b.device))
                print("A_prev type: "+str(A_prev.dtype)+" shape "+str(A_prev.shape)+" dev "+str(A_prev.device))
            # free the memory from the previous Z and A to be replaced by their new values
            self.Z = self.A = None 
            # self.mempool.free_all_blocks()
            
            self.Z = cp.dot(self.w, A_prev) + self.b
            
            # print("Z shape is: "+str(self.Z.shape)+" b shape is: "+str(self.b.shape))
            # self.Z += self.b
            self.A = self.activation.fn(self.Z)
            mems_after = check_gpu_mem(False)
            exp_added = (self.Z.nbytes + self.A.nbytes)/2**20
            if jump:
                exp_added += A_prev.nbytes/2**20
            print("size of Z and A is {:.3f} {:.3f} for total: {:.3f} size of a_prev is {:.3f}"\
                  .format(self.Z.nbytes/2**20, self.A.nbytes/2**20, exp_added, A_prev.nbytes/2**20))
            print("memory added is {:.1f} MiB and total is {:.1f} MiB on GPU{}"\
                  .format(mems_after[self.gpu][0] - mems_before[self.gpu][0],\
                          mems_after[self.gpu][0], self.gpu))
            # add dropout here and scale self.A as needed (scaling also required in backprop!)
            print("error for mem added: {:.1f}".format(mems_after[self.gpu][0] - mems_before[self.gpu][0] - exp_added))
            if jump:
                A_prev = None
            return self.A

    def backprop(self, dZ_after, w_after, A_before, m, jump_forward, jump_backward):
        with cp.cuda.Device(self.gpu):
            # mark variables as free memory before re-assigning (doesn't double memory usage)
            self.dZ = self.dw = self.db = None

            before = check_gpu_mem()            
            if jump_forward: # layer before this one is on a different GPU
                A_before_this_gpu = cp.copy(A_before)
                print("JUMPING FORWARD - A_before shape {} dev {} mem {:.3f} and after shape {} dev {} mem {:.3f}".format(A_before.shape, A_before.device, A_before.nbytes/2**20, A_before_this_gpu.shape, A_before_this_gpu.device, A_before_this_gpu.nbytes/2**20))
            else:
                A_before_this_gpu = A_before
            
            if w_after.size == 0 : # last layer in the network, dZ given based on error
                self.dZ = dZ_after
                self.dw = 1/m*cp.dot(self.dZ, A_before_this_gpu.T)
                self.db = 1/m*cp.sum(self.dZ, axis=1, keepdims=True)
                return self.dZ
                
            if jump_backward: # layer after this one is on a different GPU
                dZ_after = cp.asarray(dZ_after, dtype=cp.float32)
                w_after = cp.asarray(w_after, dtype=cp.float32)                
                print("JUMPING BACKWARD - dA shape {} dev {} mem {:.3f}".format(dA.shape, dA.device, dA.nbytes/2**20))

            after = check_gpu_mem()                
            # self.mempool.free_all_blocks()
            # self.dZ = dA * self.activation.dfn(self.Z)
            print("self.Z shape {}  mem {:.3f} dev {}".format(self.Z.shape,self.Z.nbytes/2**20,self.Z.device))

            self.dZ = cp.dot(w_after.T, dZ_after) * self.activation.dfn(self.Z)
            after2 = check_gpu_mem()
            A_before_this_gpu = A_before_this_gpu.T
            print('did transpose')
            print("devs are: self.Z {} self.dZ {}\nA_before_this_gpu {} A_before {}\n dZ_after {} w_after{}".format(self.Z.device, self.dZ.device, A_before_this_gpu.device, A_before.device, dZ_after.device, w_after.device))            
            self.dw = 1/m*cp.dot(self.dZ, A_before_this_gpu)
            self.db = 1/m*cp.sum(self.dZ, axis=1, keepdims=True)
            
            # # print("w flags")
            # # print(self.w.flags)
            # self.w = cp.ascontiguousarray(self.w.T)
            # print("w flags")
            # print(self.w.flags)

            # # self.dZ = cp.asfortranarray(self.dZ)            
            # print("dZ flags")
            # print(self.dZ.flags)
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
