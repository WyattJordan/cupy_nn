from network import network
import numpy as np

class gradcheck():
    def __init__(self,network):
        self.net = network
        self.layers = network.layers

    def check_equal(self, layers):
        equal = True
        if len(layers)!=len(self.layers):
            return False
        for sl,l in zip(self.layers, layers):
            if not (sl==l):
                equal = False
                break;
        return equal

    # Send vector of attributes to an array of layers
    def vec_to_layer_arrays(self, vec, i, attributes=["w","b"]):
        idx=0        
        for l in self.layers:
            for a in attributes:
                new_params = np.array(vec[idx: idx+getattr(l,a).size, 0]).reshape(getattr(l,a).shape)
                setattr(l, a, new_params)
                idx = idx + getattr(l,a).size
                
    # Create a column vector of all specified attributes for every layer
    def layer_arrays_to_vec(self, attributes=["w","b"]):
        vec = np.empty([0,0])
        for l in self.layers:
            for a in attributes:
                vec = np.append(vec, getattr(l,a).ravel())
        vec = np.reshape(vec, (vec.size, 1)) # column vector
        return vec

    # Manually calculates the gradient for every parameter in the model
    # So slow basically useless (must do forward propagatation 2x for
    # each parameter)
        
    # Check gradient will forward propagate many times, editing layer.(Z,A)
    # it will not backprop or edit layer.(dA, DZ, dw, db), won't interfere w/ update    
    def check_gradient(self, X, Y):
        # line below seems to not be deep copying np.arrays...
        layers_copy = [copy.deepcopy(l) for l in self.layers]
        epsilon = 1e-7
        params_vec = self.layer_arrays_to_vec()
        self.approx_grads_vec = np.zeros_like(params_vec)

        for i,param in enumerate(params_vec):
            if i%300==0:
                print("On param "+str(i)+" of "+str(params_vec.size))
            # find cost for slightly larger and slightly smaller values of param
            # approximate gradient = C+ - C- / (2*e)
            params_vec[i] = params_vec[i] + epsilon
            self.vec_to_layer_arrays(params_vec,i) # changes self.layers
            cost_plus = self.net.compute_cost(self.net.propagate_all(X),Y)
            
            params_vec[i] = params_vec[i] - 2.*epsilon
            self.vec_to_layer_arrays(params_vec, i) # changes self.layers
            cost_minus = self.net.compute_cost(self.net.propagate_all(X),Y)
            self.approx_grads_vec[i] = (cost_plus - cost_minus) / (2.*epsilon)
            
            # check we have 2 different instances of layers, one adjusted other original
            assert(self.check_equal(layers_copy) == False)
            # reset parameter that was tested and find approximate gradient
            params_vec[i] = params_vec[i] + epsilon
            self.vec_to_layer_arrays(params_vec, i)
            assert(self.check_equal(layers_copy) == True)

        grads_vec  = self.layer_arrays_to_vec(["dw","db"])
        num   = np.linalg.norm(self.approx_grads_vec - grads_vec)
        denom = np.linalg.norm(self.approx_grads_vec) + np.linalg.norm(grads_vec)
        self.grad_check = num / denom
        assert(self.layers == layers_copy) # layers should not be edited
        return self.grad_check

    def output_results():
        check_max = self.grad_check[self.grad_check>1e-7]
        print("There are "+str(check_max.size)+" gradients w/ error > 1e-7.")
        print("Gradients have idxs in model: ")
        print(np.where(check>1e-7))
        print("Gradients have values ")
        print(check_max)
