import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.asarray(x)
    if x.ndim == 3: axis = (1, 2)
    else: axis = (2, 3)
    return x.mean(axis=axis)