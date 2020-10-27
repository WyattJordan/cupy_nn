import numpy as np
import subprocess


def check_gpu_mem(output=True):
    # Returns (num gpu, 3) np array
    # columns: used memory, total memory on gpu, % used
    info = str(subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE).communicate()[0])
    gpu_fan_idxs = [i for i, ch in enumerate(info) if '%'==ch]
    gpu_mems = np.zeros([len(gpu_fan_idxs), 3])

    for k,idx in enumerate(gpu_fan_idxs):
        blocks = [i for i, ch in enumerate(info[idx:idx+80]) if '|'==ch]
        start = info.find('|',idx, idx+50)
        end =   info.find('MiB',idx, idx+50)
        gpu_mems[k][0] = int(info[start+1:end]) # used memory
        start = end+4
        end = info.find('MiB',start, start+50)
        gpu_mems[k][1] = int(info[start+1:end]) # total memory on GPU
        gpu_mems[k][2] = 100.*gpu_mems[k][0] / gpu_mems[k][1] # % used

    if output:
        for i in range(gpu_mems.shape[0]):
            print("GPU {} has used {:7} MiB out of {:7}, MiB which is {:.2f} %"\
                  .format(i,gpu_mems[i][0], gpu_mems[i][1], gpu_mems[i][2]))
    return gpu_mems
        
        
        
    
    
