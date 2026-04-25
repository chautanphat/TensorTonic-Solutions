import numpy as np

def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    T = len(log_probs)
    G = np.zeros(T)
    G[T - 1] = rewards[T - 1]
    for t in range (T- 2 , -1, -1): G[t] = rewards[t] + gamma * G[t + 1]
    mean_G = np.mean(G)
    A = G - mean_G
    L = -np.mean(log_probs * A)
    return float(L)