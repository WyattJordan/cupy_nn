import argparse
import sys
import cupy as cp

class parser:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train or test a neural network')
        parser.add_argument('session_name',
                            help='name for this session (used to save models, label plots, etc.)')
        
        parser.add_argument('--load_model', 
                            help='name of model in ../model/ if loading a pre-trained model')
        parser.add_argument('--gpu', type=int,
                            help='which GPU to use', default=1)
        parser.add_argument('--memlimit', type=int,
                            help='amount of GPU memory (MiB) to allocate to cupy', default=1950)
        
        parser.add_argument('--layers',
                            help='neurons per layer as a comma separated list (Note: last layer is already set to 10 for MNIST')
        parser.add_argument('--funcs',
                            help='activation functions per layer as a comma separated list (options: relu,sigmoid,tanh, default: relu)')
        parser.add_argument('--types',
                            help='data types for each layer as a comma separated list (options:  int32,int64,fp16,fp32,fp64, default: fp32)')
        parser.add_argument('--save_model',
                            help='whether or not to save the model after training')

        parser.add_argument('--epochs', type=int,
                            help='number of epochs to complete (training set has 50k examples)', default=2)
        parser.add_argument('--batchsz', type=int,
                            help='number of examples per mini-batch', default=256)
        
        parser.add_argument('--l_rate', type=float,
                            help='set the learning rate, default is 0.04', default=0.04)
        parser.add_argument('--output_rate', type=int,
                            help='number of mini-batches before outputting an update (cost and acc)', default=30)
        parser.add_argument('--lambd', type=float,
                            help='specify lambda for L2 regularization, defualt is 0', default=0.)
        parser.add_argument('--validate_rate', type=int,
                            help='number of epochs before running evaluation', default=10)
        parser.add_argument('-plot_epochs', type=int,
                            help='make plot of cost function and accuracy for every mini batch', default=1)
        parser.add_argument('--plot_valids', type=int,
                            help='make plot of cost function and accuracy for each validation', default=1)

        
        
        # common args that need to be specified
        #     load model path?
        #     new model shape/activations/precisions/
        #
        #
        # Args that can be defaulted and rarely used:
        #     batchsz, gpu, mempoollimit, etc

        args = parser.parse_args()

        if args.load_model:
            return args, []

        else:
            if not args.layers:
                print("either specify the desired layer architecure with -layers or load a model")
                sys.exit()

            datadict = {
                "int32" : cp.int32,
                "int64" : cp.int64,
                "fp16"  : cp.float16,
                "fp32"  : cp.float32,
                "fp64"  : cp.float64
            }
                
            # change args.<var> for each var that needs edited
            # MNIST example has 784 input features and last layer for MNIST is always 10
            args.layers = [784] + [int(l) for l in args.layers.split(",")] + [10]
            if args.funcs:
                args.funcs = [f for f in args.funcs.split(",")] + ["sigmoid"]
            else:
                args.funcs = ["relu"] * (len(args.layers)-2) + ["sigmoid"]
                # args.funcs = ["relu"] * (len(args.layers)) + ["sigmoid"]                

            if args.types:
                args.types = [datadict[t] for t in args.types] + [cp.float32]
            else:
                args.types = [cp.float32] * (len(args.layers)+1)

            print(args.layers)
            print(args.funcs)
            print(args.types)            
