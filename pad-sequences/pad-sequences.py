import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    L = (max_len if max_len != None else max(len(seq) for seq in seqs))
    for seq in seqs:
        while len(seq) < L:
            seq.append(pad_value)
        while len(seq) > L:
            seq.pop()
    return seqs