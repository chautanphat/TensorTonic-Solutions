import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    bins = [[] for i in range(n_bins)]
    n = len(y_true)
    for i in range(n):
        idx = np.floor(y_pred[i] * n_bins).astype(int)
        if idx == n_bins: idx -= 1
        bins[idx].append(i)

    ece = 0
    for bin in bins:
        if len(bin) == 0: continue
        acc = np.mean(y_true[bin])
        conf = np.mean(y_pred[bin])
        ece += len(bin)/n * np.abs(acc - conf)
        
    return ece