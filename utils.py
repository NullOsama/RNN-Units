import numpy as np

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