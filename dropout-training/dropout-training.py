import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x)
    r = rng if rng is not None else np.random
    random_vals = r.random(x.shape)
    dropout_pattern = np.where(random_vals <= p, 0, 1 / (1 - p))
    output = x * dropout_pattern
    return output, dropout_pattern