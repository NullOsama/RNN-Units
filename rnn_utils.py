import numpy as np

def softmax(x):
    """
        Implementation of the softmax function.  used the max trick so the values stay small and the exp function don't explod.

        Parameters
        ----------
        x : vector of shape (C,)

        Returns
        ----------
        numpy array
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

def update_parameters_adam(parameters, gradients, v, s, t, learning_rate=0.01, beta1=0.99, beta2=0.999, epsilon=1e-8):
    """
        Update parameters using Adam.
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2  # Number of the notwork layers
    v_corrected = {} # Moving avg. of the first gradient but with bias correction.
    s_corrected = {} # Moving avg. of the squared gradient but with bias correction.



    for l in range(L):  # This loop is from 0 to L - 1, but weights are from 1 to L so we need to use l + 1.
        # Moving avg. of the gradients.
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * gradients['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * gradients['db' + str(l + 1)]

        # Add bias correction
        v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - beta1 ** t)
        v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - beta1 ** t)

        # Moving avg. of the squared gradients
        s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * np.square(gradients['dW' + str(l + 1)])
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * np.square(gradients['db' + str(l + 1)])
    
        # Add bias correction
        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - beta2 ** t)
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - beta2 ** t)

        # Update parameters

        parameters['dW' + str(l + 1)] += -learning_rate * v_corrected['dW' + str(l + 1)] / (np.sqrt(s_corrected['dW' + str(l + 1)] + epsilon))

        parameters['db' + str(l + 1)] += -learning_rate * v_corrected['db' + str(l + 1)] / (np.sqrt(s_corrected['db' + str(l + 1)] + epsilon))
    
    return parameters, v, s

