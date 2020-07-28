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