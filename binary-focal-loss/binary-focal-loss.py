import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    p_t = [predictions[i] if targets[i] == 1 else 1 - predictions[i] for i in range(len(predictions))]
    p_t = np.array(p_t)
    FL = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)
    return np.mean(FL)