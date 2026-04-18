import numpy as np

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.asarray(X)
    H, W = X.shape
    
    H_out = (np.floor((H - pool_size) / stride) + 1).astype(int)
    W_out = (np.floor((W - pool_size) / stride) + 1).astype(int)

    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            out[i, j] = np.max(X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size])
            
    return out.tolist()