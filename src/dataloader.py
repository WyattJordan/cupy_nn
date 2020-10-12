import numpy as np
import cupy as cp
import gzip
import pickle

def get_example_size():
    x,y = load_data('test')
    size_x = x[:,0].nbytes
    size_y = y[:,0].nbytes
    return size_x, size_y
    

def load_data(setname):
    f = gzip.open('../data/mnist.pkl.gz','rb')
    train, valid, test = pickle.load(f,encoding="bytes")
    # x is (784, n) and y is (10, n) with n=50,000 for training and
    # 10,000 for both validation and test data
    if setname=='train':
        x, y = format_dataset(train)
    if setname=='validate':
        x, y = format_dataset(valid)
    if setname=='test':
        x, y = format_dataset(test )
    
    return x, y

def format_dataset(data):
    data_x = np.array(data[0],dtype=np.double).T
    data_y = np.zeros([10,len(data[1])])
    for i,y in enumerate(data[1]):
        data_y[y][i] = 1.0
    return data_x, data_y

def load_data_gpu(setname, gpu=1):
    f = gzip.open('../data/mnist.pkl.gz','rb')
    train, valid, test = pickle.load(f,encoding="bytes")
    # x is (784, n) and y is (10, n) with n=50,000 for training and
    # 10,000 for both validation and test data
    if setname=='train':
        x, y = format_dataset_gpu(train)
    if setname=='validate':
        x, y = format_dataset_gpu(valid)
    if setname=='test':
        x, y = format_dataset_gpu(test )
    
    return x, y

def format_dataset_gpu(data,gpu=1):

    data_x = np.array(data[0],dtype=cp.double).T
    data_y = np.zeros([10,len(data[1])])
    data_x = data_x[:,0:5000] # dataset was too large to fit on GPU memory for testing BGD
    data_y = data_y[:,0:5000] # slicing is a temp. fix (will add mini batches for SGD soon)
    for i,y in enumerate(data_y.astype(int)):
        data_y[y][i] = 1.0
    with cp.cuda.Device(gpu):
        data_x = cp.asarray(data_x, dtype=cp.float32)
        data_y = cp.asarray(data_y, dtype=cp.float32)
        # data_x = cp.asarray(data_x[:][0:20000], dtype=cp.float32)
        # data_y = cp.asarray(data_y[:][0:20000], dtype=cp.float32)
    return data_x, data_y
