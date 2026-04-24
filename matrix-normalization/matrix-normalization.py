import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    valid_norm_type = ['l1', 'l2', 'max']
    valid_axis = [0, 1, None]
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or norm_type not in valid_norm_type or axis not in valid_axis: return None
    eps = 10**-9
    if norm_type == 'l1': x = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == 'l2': x = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
    else: x = np.max(matrix, axis=axis, keepdims=True)
    return matrix / (x + eps)