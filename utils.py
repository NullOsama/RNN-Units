import numpy as numpy

def smooth(loss, curr_loss):
    """
        Smooth the loss using exponentialy weighted averages.
    """
    beta = 0.999
    return beta * loss + (1 - beta) * curr_loss

def print_sample(sample_idx, idx2char):
    text = ''.join(idx2char[ch] for ch in sample_idx)
    print(f'----\n {text} \n----')