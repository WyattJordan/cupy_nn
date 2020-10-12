import numpy as np
import cupy as cp
import sys
import dataloader
from layer import layer
from network import network
from plotter import plotter
from grad_check import gradcheck
from init import distribute_model, set_GPU_mems
from get_mem import check_gpu_mem

def main():
    print("Initializing...")
    np.set_printoptions(threshold=np.inf)

    # Dev 0 will allocate 1600 MiB (leaving ~60MiB free)
    # Dev 1 will allocate 1930 MiB (leaving ~70MiB free)
    mems_alloc = np.array([1500, 1930])    
    set_GPU_mems(mems_alloc)

    print("memory before loading data:")
    check_gpu_mem()
    in_sz, out_sz, x_example_sz, y_example_sz = dataloader.get_sizes()
    batch_sz = 5000 #train_x.shape[1] # whole data set for BGD
    print("in_sz: {} and out_sz: {}".format(in_sz, out_sz))
    # Assign GPUs for each layer and load the model
    dims = [in_sz, 3600, 300, out_sz]  # default mem test, 0.26 GB
    dims = [in_sz, 3600, 3000, out_sz] # expected 137MB added for 1st propagat but added 167
    dims = [in_sz, 4600, 2000, out_sz] # error of 30MB as well
    dims = [in_sz, 5600, 2000, out_sz] # error of 30MB again
    dims = [in_sz, 7600, 2000, out_sz] # error of 30MB, good total w/ 26% usage
    dims = [in_sz, 9600, 2000, out_sz] # error of 30MB, total under by 2MiB  w/ 32% usage
    dims = [in_sz, 13600, 2000, out_sz] # error of 30MB, total under by 7MiB w/ 41%
    dims = [in_sz, 17600, 2000, out_sz] # error of     total under by 11MiB  w/ 51%
    dims = [in_sz, 23600, 2000, out_sz] # error of     total under by 20MiB  w/ 65%
    dims = [in_sz, 28600, 2000, out_sz] # error of     total under by 26MiB  w/ 78%
    dims = [in_sz, 35600, 2000, out_sz] # error of     total under by 34MiB, 95% on GPU1
    dims = [in_sz, 38600, 2000, out_sz] # error of     total under by
    # isn't distributing the model...'
    dims = [in_sz, 17600, 2000, out_sz]  # total under by 11MiB  w/ 51%    
    dims = [in_sz, 13600, 4000, out_sz]  # total under by 7MiB + 4MiB, 50% usage
    dims = [in_sz, 13600, 12000, out_sz] # total under by 29MiB w/ 86% usage
    dims = [in_sz, 13600, 15000, out_sz] # total under by 31MiB w/ 91% usage

    dims = [in_sz, 30600, 1800, out_sz] # error of     total under by    

    activations = ["relu","relu","sigmoid"]
    gpus = distribute_model(dims, batch_sz, x_example_sz, y_example_sz)
    net = network(dims, activations, batch_sz, gpus)

    train_x, train_y = dataloader.load_data_gpu('train', gpu_start=gpus[0], gpu_end=gpus[-1])    
    print("--------------------------------------------------")
    print("batch sz is: "+str(batch_sz))        
    print("batch_xdata_sz is: "+str(train_x.nbytes + train_y.nbytes)+\
          " or in MiB: "+str((train_x.nbytes + train_y.nbytes)/(2**20)))
    print("bytes: "+str(train_x.nbytes))
    print("--------------------------------------------------")


    epochs = 50
    learning_rate = 0.04
    costs = []
    accs  = []
    checkgrad = False
    plot = True

    # Load training data on correct gpus.. will do this for each mini batch when SGD
    # NOT NEEDED YET (dataloader loads on gpu)
    # with cp.cuda.Device(gpus[0]):
    #     train_x = cp.asarray(train_x)
    # with cp.cuda.Device(gpus[-1]):
    #     train_y = cp.asarray(train_y)
    with cp.cuda.Device(gpus[0]):
        train_x = cp.asarray(train_x, dtype=cp.float32)
    with cp.cuda.Device(gpus[-1]):        
        train_y = cp.asarray(train_y, dtype=cp.float32)
    
    print("data types are Y: {}, and X {} and devs are: {}, {}".format(str(type(train_y)), str(type(train_x)), str(train_x.device),str(train_y.device) ))
    print("gpu_start was: "+str(gpus[0])+" gpu_end was: "+str(gpus[-1]))

    
    for e in range(epochs):
        #learning_rate = learning_rate / (1 + 0.08*e) # can decay if desired
        check_gpu_mem()
        output = net.propagate_all(train_x)
        check_gpu_mem()

        
        costs.append(net.compute_cost(train_y))
        accs.append (net.compute_acc( train_y)) 
        print("Epoch {} has acc {}% and cost {} with rate {}".format(e,accs[-1],costs[-1],learning_rate))

        net.backprop_all(train_y)
        check_gradient(net, train_x, train_y) if checkgrad else None
        net.update_all(learning_rate)

    if plot:
        plt = plotter(dims, costs, accs, learning_rate)
        plt.plot("and plain back gradient descent")

if __name__ == "__main__":
    main()
