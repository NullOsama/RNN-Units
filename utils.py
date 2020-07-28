import numpy as np
from rnn_utils import softmax

def smooth(loss, curr_loss):
    """
        Smooth the loss using exponentialy weighted averages.

        Parameters
        -----------
        loss : Last computed loss (previous loss)
        curr_loss : currently computed loss

        Returns
        ----------
        smoothed_loss : float
                            Smoothed loss across the last 1000 losses(aprox.).
    """
    beta = 0.999
    return beta * loss + (1 - beta) * curr_loss

def print_sample(sample_idx, idx2char):
    """
        Convert indices of the sample to charecters and join them to construct a sentence.
    """
    text = ''.join(idx2char[ch] for ch in sample_idx)
    print(f'----\n {text} \n----')

def get_initial_loss(vocab_size, seq_length):
    return - np.log(1.0 / vocab_size) * seq_length
    

def initialize_parameters(n_a, n_x, n_y):
    """
        Initialize parameters with small random values

        Returns:
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            b --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    Wax = np.random.randn(n_a, n_x) * 0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}

    return parameters

def rnn_step_forward(parameters, a_prev, x):
    """
        One iteration of forward pass on layer

        Parameters
        ----------
        parameters : python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        a_prev : ndarray of the activations from the previous time step

        x : ndarray of the inpute

        Returns
        ----------
        a_next : ndarray
                        activation of the next time step
        p_t : ndarray
                    y hat of the current input
    """
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(np.concatenate((Waa, Wax), axis=1), np.concatenate((a_prev, x), axis=0)) + b) # hidden state
    p_t = softmax(np.dot(Wya, a_next) + by) # unnormalized log probabilities for next chars # probabilities for next chars 
    
    return a_next, p_t



def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    """
        One iteration of backward pass on layer

        Parameters
        ----------
        dy : Gradient of the output, numpy array of shape (ny, 1)

        gradients : python dictionary containig:
                        dWax -- Gradient of the Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        dWaa -- Gradient of the Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        dWya -- Gradient of the Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        db --  Gradient of the Bias, numpy array of shape (n_a, 1)
                        dby -- Gradient of the bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        parameters : python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        x : ndarray of the inpute

        a : ndarray of the activations from the current time step

        a_prev : ndarray of the activations from the previous time step

        Returns
        ----------
        Gradients
    """

    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)

    return gradients



def update_parameters(parameters, gradients, lr):
    """
        Update the parameters of the network with Gradient descent
    """

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters