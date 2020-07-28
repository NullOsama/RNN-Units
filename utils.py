import numpy as numpy

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