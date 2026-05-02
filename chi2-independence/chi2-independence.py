import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    row_sums = np.sum(C, axis=1, keepdims=True)
    col_sums = np.sum(C, axis=0, keepdims=True)
    total_sum = np.sum(C)

    expected = (row_sums @ col_sums) / total_sum
    chi2_stat = np.sum((C - expected) ** 2 / expected)

    return chi2_stat, expected