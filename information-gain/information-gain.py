import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    n = len(y)
    lf = [y[i] for i in range(n) if split_mask[i]]
    rt = [y[i] for i in range(n) if not split_mask[i]]
    if min(len(lf), len(rt)) == 0: return 0.0
    return _entropy(y) - (len(lf)/n * _entropy(lf) + len(rt)/n * _entropy(rt))