import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    idx = dict.fromkeys(vocab, 0)
    
    i = 0
    for word in vocab:
        idx[word] = i
        i += 1

    bow = np.zeros(len(vocab), dtype=int)
    for token in tokens:
        if token in vocab: bow[idx[token]] += 1
            
    return bow