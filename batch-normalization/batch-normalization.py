import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    shape = x.shape
    if len(shape) == 2: axis = 0
    else:
        axis = (0, 2, 3)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.mean((x - mu)**2, axis=axis, keepdims=True)
    x_hat = (x - mu)/np.sqrt(sigma + eps)
    y = gamma * x_hat + beta
    return y