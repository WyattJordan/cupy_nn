import numpy as np
import cupy as cp
import itertools
import sys

def set_GPU_mems(mems):
    mempool = cp.get_default_memory_pool()
    for i,mem in enumerate(mems):
        with cp.cuda.Device(i):
            mempool.set_limit(size=mem*1024**2) # convert MiB to bytes
    
def distribute_model(dims, batch_size, x_size, y_size, num_gpu=2):
    # x_size is bytes for one example
    # y_size is bytes for one label
    # m is batch size
    
    # Returns list of GPU indexes corresponding to each layer.
    # Can have multiple jump points (locations where a[l-1] and a[l] are on different GPUs)

    # The mini-batch data and as much of the network as possible will be 
    # allocated on GPU1. Remaining layers will be allocated on GPU0.
    # This creates one jump point so activations only need to switch memory once.
    
    # Splitting up the layers for better memory optimization/fitting bigger models is possible by
    # having multiple jump points. Undetermined if that would lead to significantly
    # slower execution times (what's the time penalty for transferring activations between GPUs?)

    # Get available memory
    gpu_mem = np.zeros(num_gpu) # in MiB
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            gpu_mem[i] = cp.get_default_memory_pool().get_limit() / 2**20
            # print("Allocated Memory for GPU "+str(i)+" is {} MiB".format(gpu_mem[i]))
            
    # need 34MiB for libraries, 30MiB for operations, 40MiB for metadata w/ largest models
    print("total allocated GPU Memory is {:.3f} GiB".format(gpu_mem.sum()/1024))
    gpu_mem = gpu_mem - 34 - 30 - 40
    print("allocated GPU Memory w/ margin is {:.3f} GiB".format(gpu_mem.sum()/1024))
    
    # each column is one layer
    # rows are memory required (in MiB) for:
    # 0 - weights (self.w)
    # 1 - bias    (self.b)
    # 2 - linear units (self.Z) activations (self.A) and derivatives (self.dZ and self.dA)
    #     note - dependent on batch size but all have equal size
    # 3 - data (for input and output layer)
    # 4 - sum for this layer (assuming no jumps)
    # 5 - previous activations (self.A_prev) only occurs on forward jump
    # 6 - previous deriv and weights during backprop (self.w, self.dZ)  only occurs on backward jump
    layer_mems = np.zeros([7,len(dims)-1])
    layer_mems[3][0]  = x_size*batch_size / 2**20 # keep batch inputs and first layer on same GPU
    layer_mems[3][-1] = y_size*batch_size / 2**20 # keep batch labels and last layer on same GPU    

    layer = 0
    for i in range(1,len(dims)):
        layer_mems[0][layer] = 4 * dims[i] * dims[i-1]  / 2**20 # in MiB, assumes FP32
        layer_mems[1][layer] = 4 * dims[i] / 2**20
        layer_mems[2][layer] = layer_mems[1][layer] * batch_size * 4
        layer_mems[4][layer] = np.sum(layer_mems[:5,i-1])
        layer_mems[5][layer] = 4 * dims[i-1] * batch_size / 2**20
        if i!=len(dims)-1:
            layer_mems[6][layer] = layer_mems[2][layer+1] / 4 + layer_mems[0][layer+1]
        layer += 1
    
    # Check GPUs have enough memory
    total = np.sum(layer_mems[4,:])
    print("total is: {:.4f} GiB with {:.4f} GiB remaining"\
          .format((total+70)/1024, (gpu_mem.sum()-total)/1024))
    if gpu_mem.sum() - total < 0:
        print("Model cannot be fit into allocated GPU memory, exiting")
        sys.exit()

    # options has a column for every layer. Every element is a gpu index.
    # Each row is a unique combination of assigning each layer to a gpu
    options = list(list(tup for tup in itertools.product(range(num_gpu),repeat=len(dims)-1)))
    options = np.array(options)
    # results columns: num jumps, min GPU memory remaining, [mem used for gpu_i], [options]
    results = np.zeros([options.shape[0], 2 + num_gpu]) # row for every option
    results = np.append(results, options, axis=1)       # append options for lexsort
    print("Finding best distribution across {} GPUs: ".format(num_gpu), end='')
    gpu_mem_used = np.zeros(num_gpu)
    for i,op in enumerate(options):
        for k,gpu in enumerate(gpu_mem_used):
            gpu_mem_used[k] = np.sum((layer_mems[4,:])[(op==k).astype(bool)])
            jump_forward  = np.logical_and(op==k, np.diff(op, prepend=op[0])!=0)
            jump_backward = np.logical_and(op==k, np.append(jump_forward, [0])[1:])
            gpu_mem_used[k] += np.sum((layer_mems[5,:])[jump_forward])
            gpu_mem_used[k] += np.sum((layer_mems[6,:])[jump_backward])            
            results[i][k+2] = gpu_mem[k] - gpu_mem_used[k]
            
        results[i][0] = (np.diff(op)!=0).sum() # num jumps (require GPU mem copy)
        results[i][1] = np.amin(results[i][2:2+num_gpu])
        if results[i][1] < 0: # if a GPU runs out of mem set jumps to max+1
            results[i][0] = len(dims)

    # sort by min jumps, then by largest of minimum GPU memory left
    # could add another criteria to minimize: size of data to jump between GPUs
    results = results[np.lexsort((-results[:,1],results[:,0]))]
    gpu_idxs = results[0][2+num_gpu : 2 + num_gpu + len(dims)-1]    
    print("Layers will be on GPUs "+str(gpu_idxs)+" respectively.")
    return gpu_idxs.astype(np.int32)
