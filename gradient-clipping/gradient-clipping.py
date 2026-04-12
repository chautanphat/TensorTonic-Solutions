import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g, dtype=float)
    _g = np.linalg.norm(g)
    if _g > max_norm and max_norm > 0:
        g *= max_norm/_g
    return g