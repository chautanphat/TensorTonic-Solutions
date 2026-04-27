import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    return (np.where(np.asarray(x) == 1, p, 1 - p), p, p * (1 - p))