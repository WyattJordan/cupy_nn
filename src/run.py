import numpy as np
import cupy as cp
import pickle
import sys
from args import parser
from dataloader import dataloader
from layer import layer
from network import network
from plotter import plotter
from grad_check import gradcheck
from init import distribute_model, set_GPU_mems
from get_mem import check_gpu_mem

def main():
    np.set_printoptions(threshold=np.inf)
    args = parser()
    print(args)

    in_sz, out_sz  = 784, 10 # for MNIST dataset
    if args.load_model:
        print("Loading Entire Model....")
        with open("../models/"+ str(args.load_model)) as f:
            args.gpu, args.batchsz, args.memlimit, net = pickle.load(f)
    else:
        print("Initializing New Model....")
        mem_before_model = check_gpu_mem(False)

        # manual override section if desired over CL arguments
        args.dims = [in_sz, 8192, 4096, out_sz]
        args.activations = ["relu","relu","sigmoid"]
        args.gpus = [gpu, gpu, gpu]
        args.precision = [cp.float16, cp.float16, cp.float16]

        
        net = network(args.dims, args.activations, args.batchsz, args.gpus, args.types, args.lambd)
        mem_after_model = check_gpu_mem(False)
        print("Memory added from making model in MiB is: ")
        print(mem_after_model - mem_before_model)

    
    sys.exit()
    cp.cuda.Device(args.gpu[0]).use()
    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=args.memlimit*1024**2) # Allocate 1950 MiB
        
    print("loading data")
    dl = dataloader(args.batchsz)
    
    # Examining some examples
    # train_x, train_y, looped, _ = dl.get_next_batch('train', gpu)
    # np.set_printoptions(edgeitems=28, linewidth=1000)
    # print(str(type(cp.asnumpy(train_x[:,0]).reshape(28,-1))))
    # print(cp.asnumpy(train_x[:,0]).reshape(28,-1))
    # print("----------")
    # print(cp.asnumpy(train_x[:,1]).reshape(28,-1))
    # print("----------")
    # print(train_y[:,2])
    # print(cp.asnumpy(train_x[:,2]).reshape(28,-1))
    # print("----------")
    # print(cp.asnumpy(train_x[:,3]).reshape(28,-1))
    

    epochs = 80
    learning_rate = 0.04
    costs = []
    accs  = []
    vcosts = []
    vaccs  = []
    checkgrad = False
    output_rate = 20

    # Load training data on correct gpus.. will do this for each mini batch when SGD
    count = 0    
    for e in range(epochs):
        #learning_rate = learning_rate / (1 + 0.08*e) # can decay if desired
        if e>40:
            learning_rate = 0.01
        if e>70:
            learning_rate = 0.005

        e_start = count
        looped = False
        while looped == False:
            train_x, train_y, looped, _ = dl.get_next_batch('train', gpu)
        
            output = net.propagate_all(train_x)    
            costs.append(net.compute_cost(train_y))
            accs.append (net.compute_acc( train_y)) 

            net.backprop_all(train_y)
            check_gradient(net, train_x, train_y) if checkgrad else None
            net.update_all(learning_rate)
            count += 1
            if count%output_rate == 0:
                avg_acc = np.mean(np.array([accs[count-output_rate:count]]))
                avg_cost = np.mean(np.array([costs[count-output_rate:count]]))                
                print("Epoch {:2d} batch {:3d} batch_acc {:.4f}% avg acc {:.4f}% cost {:.4f} avg cost {:.4f} with rate {:.4f}".format(e,count,float(accs[-1]),avg_acc,float(costs[-1]), avg_cost, learning_rate))

        acc = np.mean(np.array([accs[e_start : count-1]]))
        print("After Epoch {:3d} train set accuracy is {:.5f}".format(e, acc))
        
        if e%args.valid_rate == 0:
            cost, acc = net.evaluate(dl,'valid')
            vcosts.append(cost)
            vaccs.append(acc)

    if args.save_model:
        print("saving model")
    if args.plot_epochs:
        plt = plotter(dims, costs, accs, learning_rate)
        plt.plot("training_"+args.session_name)

    if args.plot_valids:
        plt = plotter(dims, vcosts, vaccs, learning_rate)
        plt.plot("validation_"+args.session_name)

    print("Starting test")
    cost, acc = net.evaluate(dl,'test')
    print("After Epoch {:3d} test set accuracy is {:.5f}".format(e, acc))
        
if __name__ == "__main__":
    main()
