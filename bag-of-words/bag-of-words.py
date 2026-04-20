import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    bow = dict.fromkeys(vocab, 0)
    for token in tokens:
        if token in vocab: bow[token] += 1
    return np.array(list(bow.values()), dtype=int)