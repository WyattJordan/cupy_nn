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

    # Make mems, array of memory required for each layer (add space for libararies?)
    w_mems = np.zeros([len(dims)-1,1])   # memory for weights
    b_mems = np.zeros_like(w_mems)       # memory for bias
    for i in range(1,len(dims)):
        w_mems[i-1] = 4 * dims[i] * dims[i-1]  / 2**20 # in MiB, assumes FP32
        b_mems[i-1] = 4 * dims[i] * batch_size / 2**20

    dot_mems = np.zeros_like(w_mems) # each layer needs enough ram to compute dot product
    # dot_mems[0] = x_size*batch_size/2**20 * w_mems[0]
    # for i in range(1,len(w_mems)):
    #     dot_mems[i] = w_mems[i] * b_mems[i-1]

    mems = dot_mems + w_mems + b_mems*4      # b_mems is the same for bias, Z, A, and A_prev
    mems[0][0]  += x_size*batch_size / 2**20 # keep batch inputs and first layer on same GPU
    mems[0][-1] += y_size*batch_size / 2**20 # keep batch labels and last layer on same GPU    
    print("%%%%%%%%%%%%%%%%%%%%%%% Model Mems are %%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("w_mems           b_mems           dot_mems           mems")
    print(np.concatenate((w_mems,b_mems, dot_mems, mems), axis=1))

    # Check GPUs have enough memory
    total = np.sum(mems)
    print("total is: {:.2f} GiB with {:.2f} GiB remaining"\
          .format(total/1024, (gpu_mem.sum()-total)/1024))
    if gpu_mem.sum() - total < 50:
        print("Model cannot be fit into allocated GPU memory, exiting")
        sys.exit()
        
    options = list(list(tup for tup in itertools.product(range(num_gpu),repeat=len(mems))))
    options = np.array(options)
    # results columns: crossovers, min GPU memory remaining, [mem used for gpu_i], [options]
    results = np.zeros([options.shape[0], 2 + num_gpu])
    results = np.append(results, options, axis=1)
    print("Finding best distribution across {} GPUs: ".format(num_gpu), end='')
    gpu_mem_used = np.zeros(num_gpu)
    for i,op in enumerate(options):
        for k,gpu in enumerate(gpu_mem_used):
            gpu_mem_used[k] = np.sum(mems[(op==k).astype(bool)])
            results[i][k+2] = gpu_mem[k] - gpu_mem_used[k]/(2**20)
        results[i][0] = (np.diff(op)!=0).sum() # crossovers (require GPU mem copy)
        results[i][1] = np.amin(results[i][2:2+num_gpu])
        if results[i][2:2+num_gpu].any()<0: # if a GPU runs out of mem set crossovers to max
            results[i][0] = len(mems)*1.1
            
    # sort by min crossovers, then by smallest GPU memory left (inverted)
    # could add another criteria to minimize: size of data to be crossed over
    results = results[np.lexsort((-results[:,1],results[:,0]))]
    gpu_idxs = results[0][2+num_gpu : 2 + num_gpu + len(mems)]    
    print("Layers will be on GPUs "+str(gpu_idxs)+" respectively.")
    return gpu_idxs

