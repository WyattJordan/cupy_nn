import numpy as np
import cupy as cp
import gzip
import pickle

class dataloader:
    def __init__(self, batchsz=64):
        np.random.seed(1)   # consistent randomization = debuggable network        
        self.f = gzip.open('../data/mnist.pkl.gz','rb')
        train, valid, test = pickle.load(self.f,encoding="bytes")        
        # train = self.load_data('train')
        # valid = self.load_data('valid')
        # test  = self.load_data('test')
        self.rawdata = {
            'train': train,
            'valid': valid,
            'test' : test,
        }
        self.current = {
            'train': 0,
            'valid': 0,
            'test' : 0,
        }
        self.batchsz = batchsz
        self.curr_batch = 0
        self.train_sz = 50000
        self.other_sz = 10000
        self.train_idxs  = np.random.permutation(self.train_sz)

    def get_next_batch(self, dataset, gpu=-1):
        loop  = False
        start = self.curr_batch * self.batchsz
        if dataset == 'train':
            max_sz = self.train_sz
        else:
            max_sz = self.other_sz
        end   = min((self.curr_batch+1) * self.batchsz, max_sz)
        self.curr_batch += 1
        
        if dataset=='train':
            idxs   = self.train_idxs[start:end]
        else:
            idxs   = np.array(range(start, end), dtype=np.int_)

        if len(idxs) < self.batchsz or end>=max_sz:
            if dataset=='train':
                supplement_idxs = np.random.permutation(start-1)[:(self.batchsz - len(idxs))]
                idxs = np.append(idxs, supplement_idxs)
                self.train_idxs = np.random.permutation(50000)
            self.curr_batch = 0
            loop = True            
            
        x, y = self.get_batch(self.rawdata[dataset], idxs)
        x_gpu, y_gpu = self.convert_to_gpu(gpu, x, y)
        return x_gpu, y_gpu, loop, len(idxs)
        
    # def get_next_train_batch(self,  gpu=-1, auto_fill=True):
    #     loop = False
    #     start  = self.train_batch * self.batchsz
    #     end    =(self.train_batch+1) * self.batchsz
    #     idxs   = self.train_idxs[start:end]
    #     self.train_batch += 1
        
    #     if len(idxs) != self.batchsz or end>=self.train_sz:
    #         supplement_idxs = np.random.permutation(start-1)[:(self.batchsz - len(idxs))]
    #         idxs = np.append(idxs, supplement_idxs)
    #         self.train_idxs = np.random.permutation(50000)
    #         self.train_batch = 0
    #         loop = True            
            
    #     x, y = self.get_batch(self.rawdata['train'], idxs)
    #     x_gpu, y_gpu = self.convert_to_gpu(gpu, x, y)
        
    #     return x_gpu, y_gpu, loop
    
    def convert_to_gpu(self, gpu, x, y):
        if gpu != -1:
            with cp.cuda.Device(gpu):
                x_gpu = cp.asarray(x, dtype=cp.float16)
                y_gpu = cp.asarray(y, dtype=cp.float16)
        return x_gpu, y_gpu
        
    def load_data(self, setname):
        train, valid, test = pickle.load(self.f,encoding="bytes")
        # x is (784, n) and y is (10, n) with n=50,000 for training and
        # 10,000 for both validation and test data
        if setname=='train':
            x, y = self.format_dataset(train)
        if setname=='valid':
            x, y = self.format_dataset(valid)
        if setname=='test':
            x, y  = self.format_dataset(test )
        return [x, y]

    def get_batch(self, data, idxs):
        # print("data[0] type is: "+str(type(data[0])))
        # print("data[0] shapeis: "+str(data[0].shape))                
        data_x = np.array(data[0][idxs,:],dtype=np.float32).T
        data_y = np.zeros([10,len(idxs)])
        for i,y in enumerate(data[1][idxs]):
            data_y[y][i] = 1.0        # one-hot encoding
        return data_x, data_y
    
# def get_sizes():
#     x,y = load_data('test')
#     size_x = x[:,0].nbytes
#     size_y = y[:,0].nbytes
#     x, y = load_data('test')
#     in_size = x.shape[0]
#     out_size = y.shape[0]    
#     return in_size, out_size, size_x, size_y
    
# def format_dataset_gpu(data, gpu_start=1, gpu_end=1):

#     data_x = np.array(data[0],dtype=cp.float32).T
#     data_y = np.zeros([10,len(data[1])])
#     data_x = data_x[:,0:5000] # dataset was too large to fit on GPU memory for testing BGD
#     data_y = data_y[:,0:5000] # slicing is a temp. fix (will add mini batches for SGD soon)
#     for i,y in enumerate(data_y.astype(int)):
#         data_y[y][i] = 1.0
#     with cp.cuda.Device(gpu_start):
#         data_x = cp.asarray(data_x, dtype=cp.float32)
#     with cp.cuda.Device(gpu_end):        
#         data_y = cp.asarray(data_y, dtype=cp.float32)
#         # data_x = cp.asarray(data_x[:][0:20000], dtype=cp.float32)
#         # data_y = cp.asarray(data_y[:][0:20000], dtype=cp.float32)
#     return data_x, data_y
