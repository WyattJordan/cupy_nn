import numpy as np
import cupy as cp
import itertools
import sys

def set_GPU_mems(mems):
    mempool = cp.get_default_memory_pool()
    for i,mem in enumerate(mems):
        with cp.cuda.Device(i):
            mempool.set_limit(size=mem*1024**2)
    
def distribute_model(dims, batch_size, x_size, y_size, num_gpu=2):
    # x_size is bytes for one example
    # y_size is bytes for one label
    # m is batch size
    
    # Returns list of GPU indexes corresponding to each layer.
    # Can have multiple cross-over points (locations where a[l-1] and a[l] are on different GPUs)

    # The mini-batch data and as much of the network as possible will be 
    # allocated on GPU1. Remaining layers will be allocated on GPU0.
    # This creates one cross-over point so activations only need to switch memory once.
    
    # Splitting up the layers for better memory optimization/fitting bigger models is possible by
    # having multiple cross-over points. Undetermined if that would lead to significantly
    # slower execution times (what's the time penalty for transferring activations between GPUs?)

    # Get available memory
    gpu_mem = np.zeros(num_gpu) # in MiB
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            gpu_mem[i] = cp.get_default_memory_pool().get_limit() / 2**20
            print("Allocated Memory for GPU "+str(i)+" is {} MiB".format(gpu_mem[i]))  
    print("total allocated GPU Memory is {:.3f} GiB".format(gpu_mem.sum()/1024))

    # each column is one layer
    # rows are memory required (in MiB) for:
    # 0 - weights (self.w)
    # 1 - bias    (self.b)
    # 2 - linear units (self.Z) and activations (self.A) - dependent on batch size, equal size
    # 3 - data (for input and output layer)
    # 4 - sum for this layer (assuming no crossovers
    # 5 - previous activations (A_prev) - only occurs on crossover    
    layer_mems = np.zeros([6,len(dims)-1])
    layer_mems[3][0]  = x_size*batch_size / 2**20 # keep batch inputs and first layer on same GPU
    layer_mems[3][-1] = y_size*batch_size / 2**20 # keep batch labels and last layer on same GPU    
    
    for i in range(1,len(dims)):
        layer_mems[0][i-1] = 4 * dims[i] * dims[i-1]  / 2**20 # in MiB, assumes FP32
        layer_mems[1][i-1] = 4 * dims[i] / 2**20
        layer_mems[2][i-1] = layer_mems[1][i-1] * batch_size * 2
        layer_mems[4][i-1] = np.sum(layer_mems[:5,i-1])
        layer_mems[5][i-1] = 4 * dims[i-1] * batch_size / 2**20

    print("%%%%%%%%%%%%%%%%%%%%%%% Model Mems are %%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(layer_mems)

    # Check GPUs have enough memory
    total = np.sum(layer_mems[4,:])
    print("total is: {:.4f} GiB with {:.4f} GiB remaining"\
          .format((total+70)/1024, (gpu_mem.sum()-total)/1024))
    if gpu_mem.sum() - total < 20:
        print("Model cannot be fit into allocated GPU memory, exiting")
        sys.exit()

    # options has a column for every layer. Every element is a gpu index.
    # Each row is a unique combination of assigning each layer to a gpu
    options = list(list(tup for tup in itertools.product(range(num_gpu),repeat=len(dims)-1)))
    options = np.array(options)
    # results columns: crossovers, min GPU memory remaining, [mem used for gpu_i], [options]
    results = np.zeros([options.shape[0], 2 + num_gpu]) # row for every option
    results = np.append(results, options, axis=1)       # append options for lexsort
    print("Finding best distribution across {} GPUs: ".format(num_gpu), end='')
    gpu_mem_used = np.zeros(num_gpu)
    for i,op in enumerate(options):
        for k,gpu in enumerate(gpu_mem_used):
            gpu_mem_used[k] = np.sum((layer_mems[4,:])[(op==k).astype(bool)])
            # add A_prev if there is a crossover:
            cross_cond = np.logical_and(op==k, np.diff(op, prepend=op[0])!=0)
            gpu_mem_used[k] += np.sum((layer_mems[5,:])[cross_cond])
            results[i][k+2] = gpu_mem[k] - gpu_mem_used[k]
            
        results[i][0] = (np.diff(op)!=0).sum() # num crossovers (require GPU mem copy)
        results[i][1] = np.amin(results[i][2:2+num_gpu])
        if results[i][2:2+num_gpu].any()<0: # if a GPU runs out of mem set crossovers to max+1
            results[i][0] = len(dims)
            
    # sort by min crossovers, then by largest of minimum GPU memory left
    # could add another criteria to minimize: size of data to be crossed over
    results = results[np.lexsort((-results[:,1],results[:,0]))]
    gpu_idxs = results[0][2+num_gpu : 2 + num_gpu + len(dims)-1]    
    print("Layers will be on GPUs "+str(gpu_idxs)+" respectively.")
    return gpu_idxs
