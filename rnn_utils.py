import numpy as np



def softmax(x):
    """
        Implementation of the softmax function.  used the max trick so the values stay small and the exp function don't explod.

        Parameters
        ----------
        x : vector of shape (C,)

        Returns
        ----------
        numpy.ndarray
                        Softmax of the input vector so all values are less than one and sum up to one.
    """
    x_max = np.max(x)
    return np.exp(x - x_max) / np.sum(np.exp(x - x_max))



def sigmoid(x):
    """
        Implementation of the softmax function.
    """
    return 1.0 / (1 + np.exp(-x))


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing the parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    L = len(parameters) // 2 # Number of layers in teh network

    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)        
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        
    return v, s

