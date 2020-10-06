import numpy as np
import cupy as cp
import sys
import dataloader
import itertools
from layer import layer
from network import network
from plotter import plotter
from grad_check import gradcheck

def distribute_model(mini_data_size, dims):
    # Returns list of GPU indexes corresponding to each layer.
    # Can have multiple cross-over points (locations where a[l-1] and a[l] are on different GPUs)

    # The mini-batch data and as much of the network as possible will be 
    # allocated on GPU1. Remaining layers will be allocated on GPU0.
    # This creates one cross-over point so activations only need to switch memory once.
    
    # Splitting up the layers for better memory optimization/fitting bigger models is possible by
    # having multiple cross-over points. Undetermined if that would lead to significantly
    # slower execution times (what's the time penalty for transferring activations between GPUs?)

    # Future Improvements
    # Make agnostic to number of GPUs
    
    # Get available memory
    with cp.cuda.Device(0):
        mem0_avail = cp.get_default_memory_pool().get_limit() / 2**20 # in MiB
    with cp.cuda.Device(1):
        mem1_avail = cp.get_default_memory_pool().get_limit() / 2**20
    mem_avail = mem1_avail + mem0_avail
    print("mem0_avail: {}, mem1_avail: {}, mem_avail: {} MiB"\
          .format(mem0_avail, mem1_avail, mem_avail))

    # Make mems, array of memory required for each layer
    mems = np.zeros([len(dims)-1,1])  # add amount of space needed for libraries? how much?
    mems[0][0] = mems[0][0] + mini_data_size # keep mini-batch and first layer on same GPU 
    for i in range(1,len(dims)):
        mems[i-1] = 4 * (dims[i-1]*dims[i] + dims[i]) # assumes FP32

    # Check Model can fit on the GPUs
    total = np.sum(mems)/2**20
    print("total is: {:.2f} MiB with {:.2f} MiB remaining".format(total, mem_avail-total))
    if mem_avail - total < 0:
        print("Model cannot be fit into allocated GPU memory, exiting")
        sys.exit()
        
    options = list(list(tup for tup in itertools.product(range(2),repeat=len(mems))))
    options = np.array(options)
    # results columns: crossovers, mem0 left, mem1 left, mem worst case, corresponding option
    results = np.zeros([options.shape[0],4])
    results = np.append(results, options, axis=1)
    print("Finding best method to distribute model across 2 GPUs")
    for i,op in enumerate(options):
        mem0 = np.sum(mems[(1-op).astype(bool)])
        mem1 = np.sum(mems[op.astype(bool)])
        results[i][0] = (np.diff(op)!=0).sum() # crossovers (require GPU mem copy)
        results[i][1] = mem0_avail - (mem0.sum() / (2**20)) # in MiB
        results[i][2] = mem1_avail - (mem1.sum() / (2**20))
        results[i][3] = min(results[i][1], results[i][2])
        if results[i][1]<0 or results[i][2]<0: # if a GPU runs out of mem set crossovers to max
            results[i][0] = len(mems)*1.1
            
    # sort by min crossovers, then by smallest GPU memory left (inverted)
    # could add another criteria to minimize: size of data to be crossed over
    results = results[np.lexsort((-results[:,3],results[:,0]))]
    print("chosen configuration is: ")
    print(results[0][:])
    gpu_idxs = results[0][4:4+len(mems)]
    print(gpu_idxs)
        

def main():
    print("starting...")
    np.set_printoptions(threshold=np.inf)

    mempool = cp.get_default_memory_pool()    
    with cp.cuda.Device(0):
        mempool.set_limit(size=1500*1024**2) # Dev 0 has 1500 MiB allocated (~160MiB free)
    with cp.cuda.Device(1):
        mempool.set_limit(size=1830*1024**2) # Dev 1 has 1830 MiB allocated (~170MiB free)
    
    # load data onto CPU, mini-batches will be loaded onto gpu
    train_x, train_y = dataloader.load_data('train')
    valid_x, valid_y = dataloader.load_data('validate')

    mini = 128 # 2 mini batches in gpu memory, one for current propagation, other for loading
    mini_data_size = 2*128 * train_x[:][0].size*4
    # play around with # of layers, # of hidden units, activation types
    dims = [train_x.shape[0], 1568, 392, 98, 10]
    distribute_model(mini_data_size, dims)

    dims = [train_x.shape[0], 156800, 1568, 9800, 16000]
    distribute_model(mini_data_size, dims)

    sys.exit()    

    activations = ["relu","relu","relu","sigmoid"]
    net = network(dims, activations, train_x.shape[1])
    
    epochs = 50
    learning_rate = 0.04
    costs = []
    accs  = []
    checkgrad = False
    plot = True
    for e in range(epochs):
        #learning_rate = learning_rate / (1 + 0.08*e) # can decay if desired
        output = net.propagate_all(train_x)
        costs.append(net.compute_cost(output, train_y))
        accs.append (net.compute_acc( output, train_y))
        print("Epoch {} has acc {:.3f}% and cost {:.5f} with rate {:.4f}".format(e,accs[-1],costs[-1],learning_rate))

        net.backprop_all(output, train_y)
        if checkgrad:
            check_gradient(net) # uselessly slow
        net.update_all(learning_rate)

    if plot:
        plt = plotter(dims, costs, accs, learning_rate)
        plt.plot("and plain back gradient descent")

def check_gradient(net, train_x, train_y):
    print("---------- Starting gradient check... ----------")
    gcheck = gradcheck(net)
    check = gcheck.check_gradient(train_x, train_y) # requires dw and db from backprop
    gcheck.output_results()

if __name__ == "__main__":
    main()
