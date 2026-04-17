import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if num_classes == None: num_classes = max(y_true) + 1
    k = num_classes
    eps = 10**-9

    indices = y_true * k + y_pred
    
    confusion_matrix = np.bincount(indices.astype(int), minlength=k**2).reshape(k, k)
    if normalize == 'true':
        x = confusion_matrix.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        x = confusion_matrix.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        x = confusion_matrix.sum()
    else:
        return confusion_matrix
    return confusion_matrix/(x + eps)