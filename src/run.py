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
    mems_alloc = np.array([1600, 1930]) - 120 #70 # ~70MiB needed for cupy on every GPU!
    set_GPU_mems(mems_alloc)

    print("memory before loading data:")
    check_gpu_mem()
    x_example_size, y_example_size = dataloader.get_example_size()
    batch_size = 5000 #train_x.shape[1] # whole data set for BGD

    train_x, train_y = dataloader.load_data_gpu('train')    
    print("--------------------------------------------------")
    print("batch size is: "+str(batch_size))        
    print("batch_xdata_size is: "+str(train_x.nbytes + train_y.nbytes)+\
          " or in MiB: "+str((train_x.nbytes + train_y.nbytes)/(2**20)))
    print("bytes: "+str(train_x.nbytes))
    print("--------------------------------------------------")

    # Assign GPUs for each layer and load the model
    dims = [train_x.shape[0], 3600, 300, 10]  # default mem test, 0.26 GB
    dims = [train_x.shape[0], 3600, 3000, 10] # expected 137MB added for 1st propagat but added 167
    dims = [train_x.shape[0], 4600, 2000, 10] # error of 30MB as well
    dims = [train_x.shape[0], 5600, 2000, 10] # error of 30MB again
    dims = [train_x.shape[0], 7600, 2000, 10] # error of 30MB, good total w/ 26% usage
    dims = [train_x.shape[0], 9600, 2000, 10] # error of 30MB, total under by 2MiB  w/ 32% usage
    dims = [train_x.shape[0], 13600, 2000, 10] # error of 30MB, total under by 7MiB w/ 41%
    dims = [train_x.shape[0], 17600, 2000, 10] # error of     total under by 11MiB  w/ 51%
    dims = [train_x.shape[0], 23600, 2000, 10] # error of     total under by 20MiB  w/ 65%
    dims = [train_x.shape[0], 28600, 2000, 10] # error of     total under by 26MiB  w/ 78%
    dims = [train_x.shape[0], 35600, 2000, 10] # error of     total under by 34MiB, 95% on GPU1
    dims = [train_x.shape[0], 38600, 2000, 10] # error of     total under by
    # isn't distributing the model...'
    dims = [train_x.shape[0], 17600, 2000, 10]  # total under by 11MiB  w/ 51%    
    dims = [train_x.shape[0], 13600, 4000, 10]  # total under by 7MiB + 4MiB, 50% usage
    dims = [train_x.shape[0], 13600, 12000, 10] # total under by 29MiB w/ 86% usage
    dims = [train_x.shape[0], 13600, 13000, 10] # total under by 31MiB w/ 91% usage

    
    # dims = [train_x.shape[0], 156800, 1568, 9800, 16000] # gets split between 2 GPU
    activations = ["relu","relu","sigmoid"]
    gpus = distribute_model(dims, batch_size, x_example_size, y_example_size)

    net = network(dims, activations, gpus, batch_size)
    
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
    print("data types are Y: {}, and X {}".format(str(type(train_y)), str(type(train_x))))        
    for e in range(epochs):
        #learning_rate = learning_rate / (1 + 0.08*e) # can decay if desired
        check_gpu_mem()
        output = net.propagate_all(train_x)
        check_gpu_mem()
        sys.exit()
        
        costs.append(net.compute_cost(train_y))
        accs.append (net.compute_acc( train_y)) 
        print("Epoch {} has acc {:.3f}% and cost {:.5f} with rate {:.4f}"\
              .format(e,accs[-1],costs[-1],learning_rate))

        net.backprop_all(train_y)
        check_gradient(net, train_x, train_y) if checkgrad else None
        net.update_all(learning_rate)

    if plot:
        plt = plotter(dims, costs, accs, learning_rate)
        plt.plot("and plain back gradient descent")

if __name__ == "__main__":
    main()
