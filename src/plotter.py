import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

class plotter():
    def __init__(self, dims, costs, accs, learning_rate):
        self.dims = dims
        self.costs = costs
        self.accs = accs
        self.rate = learning_rate
        
    def plot(self, desc):
        x = np.arange(0,len(self.costs))
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epochs')
        
        color = 'tab:red'        
        ax1.set_ylabel('cost')
        ax1.plot(x, self.costs, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Training Accuracy')
        ax2.plot(x, self.accs, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        #fig.tight_layout()
        
        net_str = ""
        for i in range(1,len(self.dims)):
            net_str += "L"+str(i)+"_"+str(self.dims[i])
            if i!=len(self.dims)-1:
                net_str += "-->"
        title = str(len(self.accs))+" epochs w/ rate "+str(self.rate)+" for net "+net_str+" " + desc
        plt.title("\n".join(wrap(title,80)))
        plt.savefig("../img/"+net_str+"_e:"+str(len(self.accs))+"_a:"+str(self.rate)+"_acc:"+format(self.accs[-1],'.1f')+"%.png", orientation='landscape', bbox_inches='tight')
        plt.show()
