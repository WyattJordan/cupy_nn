import cupy as cp
import numpy as np
import os
import dataloader

if __name__=="__main__":
    mempool = cp.get_default_memory_pool()    
    with cp.cuda.Device(0):
        mempool.set_limit(size=1500*1024**2) # Dev 0 has 1500 MiB allocated (~160MiB free)
    with cp.cuda.Device(1):
        mempool.set_limit(size=1830*1024**2) # Dev 1 has 1830 MiB allocated (~170MiB free)

    # dev = cupy.cuda.Device(gpu_idx)
    # get current device:  dev = cp.cuda.Device()
    # dev.id, dev.compute_capability etc
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Device.html
    
    # switching memory notes:
    # var.get() puts var on CPU
    # cp.asarray(var, dtype=float32) puts var on currently selected device

    # using a stream lets it run asynchronously in the background!!

    # checking memory
    # calculate memory usage from model, then split over 2 gpus and make sure there
    # is room for 2 mini batches to be saved on one GPU (factor mini batch mem into splitting)
    
    # splitting model over both GPUs
    # at some point the activations from one layer will be sent to the other gpu
    # when computing check that a[l-1].device == w[l].device
    # and if it doesn't make a[l-1]_copy with device = w[l].device
    # must do the same thing during backprop!!!

    cp.cuda.Device(0).use()
    print("memory limit on dev 0 is: "+str(cp.get_default_memory_pool().get_limit()))
    cp.cuda.Device(1).use()
    print("memory limit on dev 1 is: "+str(cp.get_default_memory_pool().get_limit()))
    
    print("current memory on GPUs is:")
    os.system("nvidia-smi")
    print("loading training data onto gpu 1...")
    with cp.cuda.Device(1):
        train_x_1, train_y_1, = dataloader.load_data('train')
    print("new memory on GPUs is:")
    os.system("nvidia-smi")

    print("loading training data onto gpu 0...")
    with cp.cuda.Device(0):
        train_x_0, train_y_0, = dataloader.load_data('train')
    print("new memory on GPUs is:")
    os.system("nvidia-smi")

    a = cp.array([1,3,4,5,6,3,2])
    b = cp.asnumpy(a)
    where = cp.get_array_module(a,b)
    print("arrays are on: "+str(where))
    c=a*b
    print("what happened")
    
    
    
