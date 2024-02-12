# Imports
import numpy as np
from PIL import Image
import time
from util import *
from IPython.display import clear_output
    

class HopNetBuilder:
    
    def __init__(self, size: int = None, dims: np.ndarray = None, context_dims: int | np.ndarray = None, verbose = True, interpolate = False):
        self.dims = self.__set_dims(size, dims)
        self.memory = [] # List of memory matrices
        self.context_dims = self.__set_context_dims(context_dims)
        self.verbose = verbose
        self.interpolate = interpolate
        
    def memory_add(self, matrix: np.ndarray):
        # Check if matrix is binary
        new_memory = matrix
        if not np.all(np.logical_or(new_memory == 1, new_memory == -1)):
            raise ValueError("Memory must be a binary matrix of 1 and -1")
        
        if self.dims is None:
            self.__print_v("Setting dimensions to {dims} of the added memory".format(dims = new_memory.shape))
            self.dims = np.array(new_memory.shape)
        
        # Interpolation
        if np.array_equal(np.array(new_memory.shape), self.dims):
            if self.interpolate:
                new_memory = resample_with_interpolation(new_memory, self.dims)
        else:
            raise ValueError("Provided matrix have wrong dimensions")
        self.memory.append(new_memory)
    
    def memory_show(self, index: int = None):
        if index == None:
            print("There are a total of {num} memories".format(num = len(self.memory)))
            for index_, memory in enumerate(self.memory):
                print("Memory {i}".format(i = index_))
                print_array(memory)
        else:
            print_array(self.memory[index])
    
    def memory_delete(self, index: int):
        del self.memory[index]

    def build(self, init_state: np.ndarray = None, show_steps = True, verbose = True):
        return HopNet(self.dims, self.__get_weights(), init_state = init_state, context_dims = self.context_dims, show_steps = show_steps, verbose = verbose)
    
    def memory_clear(self):
        self.memory = []
    
    def __set_dims(self, size, dims):
        match (size, dims):
            case (None, None):
                return None
            case (size, None):
                return np.array([size, size])
            case (None, dims):
                return dims
            case _:
                raise ValueError("Must provide either size or dims, not both")
    
    def __set_context_dims(self, context_dims):
        match context_dims:
            case None:
                return None
            case k if isinstance(k, int):
                return np.array([context_dims, context_dims])
            case np.ndarray:
                if context_dims.shape == (2,2):
                    return context_dims
                else:
                    raise ValueError("Context dimensions must be 2x2")
            case _:
                raise ValueError("Context dimensions must be an integer or a 2x2 matrix")
    
    def __get_weights(self):
        if self.dims is None:
            raise ValueError("No memories or dimensions provided")
        num_vertices = np.array(self.dims).prod()
        weights = np.zeros((num_vertices, num_vertices))
        for memory in self.memory:
            weights += np.outer(memory, memory)
        weights = weights / len(self.memory)
        # weights = weights - np.ones(weights.shape)
        weights = weights - np.diag(weights) # Set diagonal to 0
        return weights
    
    def __print_v(self, msg):
        if self.verbose:
            print(msg)

class HopNet:
    
    def __init__(self, dims: np.array, weights: np.ndarray | str = "random", init_state: np.ndarray | str = "random", context_dims: int | np.ndarray = None, show_steps = True, verbose = True):   
        # Excemption testing
        if weights != "random" and int(np.array(dims).prod()) != weights.shape[0]: # We assign weights between "all pairs" of vertices
            raise ValueError("Mismatch between number of vertices and edges")
        if init_state != "random" and dims.shape != init_state.shape:
            raise ValueError("Mismatch between number of vertices and initial state")
        # Assigning values
        self.dims = dims
        self.weights = self.__set_weights(weights)
        self.state = self.__set_state(init_state)
        self.context_window = self.__set_context_dims(context_dims)
        self.verbose = verbose
        self.show_steps = show_steps
    
    def show_state(self):
        print_array(self.state)
    
    def set_state(self, state: np.ndarray | str = "random"):
        self.state = self.__set_state(state)
        
    def update(self, wait = 0.1, max_iter = 10000):
        prev_state = None
        for i in range(max_iter):
            clear_output()
            prev_state = np.array(self.state) # Deep copy
            self.update_async(wait = wait)
            if self.show_steps:
                self.show_state()
            if np.array_equal(self.state,prev_state):
                clear_output()
                self.__print_v("Convergence Reached at iteration {i}".format(i = i-1))
                self.show_state()
                break 
            time.sleep(wait)

    def update_async(self, wait = 0.1):
        prev_state = self.state
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                self.state[i,j] = self.__updated_neuron_value(i,j)
                if self.show_steps:
                    self.show_state()
                    time.sleep(wait)
                    clear_output()
    
    def __updated_neuron_value(self, this_i, this_j):
        this_vertex_i = this_i * self.dims[1] + this_j
        
        start_i = max(0, this_i - self.context_window[0][0])
        end_i = min(self.state.shape[0], this_i + self.context_window[0][0] + 1)
        start_j = max(0, this_j - self.context_window[1][0])
        end_j = min(self.state.shape[1], this_j + self.context_window[1][1] + 1)

        sum = 0
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                if i == this_i and j == this_j:
                    continue
                input_vertex_j = i * self.dims[1] + j
                sum += self.weights[this_vertex_i,input_vertex_j] * self.state[i,j]
        if sum > 0:
            return 1
        else:
            return -1
        
    
    def __print_v(self, msg):
        if self.verbose:
            print(msg)
    
    # Random Weights if not initizalized
    def __set_weights(self, weights):
        if isinstance(weights, np.ndarray):
            if weights.shape[0] != np.array(self.dims).prod():
                raise ValueError("Mismatch between number of vertices and edges")
            return weights
        elif weights == "random":
            num_vertices = np.array(self.dims).prod()
            weights = np.random.rand((num_vertices, num_vertices))
        else:
            raise ValueError("Provided Weights must be a numpy array")
    
    # Random start State if not initizalized
    def __set_state(self, init_state):
        if isinstance(init_state, np.ndarray):
            if init_state.shape != self.dims:
                raise ValueError("Mismatch between number of vertices and initial state")
            return init_state
        elif init_state == "random":
            state = np.random.randint(0,2,self.dims)
            state[state <= 0] = -1 # Change all 0s to -1s
            return state
        raise ValueError("Provided initial state must be a numpy array")
    
    
    # Yes, I did not check for negative values of context_dims, but its fine, 
    # If negative values are used for, say, the left bound, it just means that the context window will completely reside on the right side of the neuron
    # We trim the context window in the update function to prevent oob anyways.
    def __set_context_dims(self, context_dims):
        ctxt_dims = context_dims
        if context_dims is None:
            ctxt_dims = np.array([[self.dims[0],self.dims[0]], [self.dims[1],self.dims[1]]])
        elif isinstance(ctxt_dims, int):
            ctxt_dims = np.array([[ctxt_dims, ctxt_dims],[ctxt_dims, ctxt_dims]])
        elif ctxt_dims.shape == (2,2) and np.all(ctxt_dims > 0):
            ctxt_dims[0][0] = min(ctxt_dims[0][0], self.dims[0])
            ctxt_dims[0][1] = min(ctxt_dims[0][1], self.dims[0])
            ctxt_dims[1][0] = min(ctxt_dims[1][0], self.dims[1])
            ctxt_dims[1][1] = min(ctxt_dims[1][1], self.dims[1])
        else:
            raise ValueError("Context dimensions must be an integer or a positive 2x2 matrix")
        return ctxt_dims