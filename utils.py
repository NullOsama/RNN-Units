import numpy as numpy

def smooth(loss, curr_loss):
    """
        Smooth the loss using exponentialy weighted averages.
    """
    beta = 0.999
    return beta * loss + (1 - beta) * curr_l