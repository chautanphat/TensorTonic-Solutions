import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    image = np.array(image)
    kernel = np.array(kernel)
    
    h, w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    padded_image = np.zeros((h + padding * 2, w + padding * 2))
    padded_image[padding:padding+h, padding:padding+w] = image
    
    h_out = np.floor((h + 2 * padding - kernel_h) / stride).astype(int) + 1
    w_out = np.floor((w + 2 * padding - kernel_w) / stride).astype(int) + 1
    
    conv_image = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            istride, jstride = i * stride, j * stride
            conv_image[i, j] = (padded_image[istride:istride+kernel_h, jstride:jstride+kernel_w] * kernel).sum()
            
    return conv_image.tolist()