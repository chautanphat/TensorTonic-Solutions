import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    
    Parameters:
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    
    Returns:
    float: The mean silhouette score across all samples.
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0.0

    X_sq_norms = np.sum(X**2, axis=1)
    dist_sq = X_sq_norms[:, np.newaxis] + X_sq_norms - 2 * np.dot(X, X.T)
    dist_matrix = np.sqrt(np.maximum(dist_sq, 0))
    cluster_distances = np.zeros((n_samples, n_clusters))
    for k, label in enumerate(unique_labels):
        mask = (labels == label)
        cluster_distances[:, k] = np.mean(dist_matrix[:, mask], axis=1)

    label_indices = np.searchsorted(unique_labels, labels)
    cluster_counts = np.bincount(label_indices)
    n_in = cluster_counts[label_indices]

    a_with_self = cluster_distances[np.arange(n_samples), label_indices]
    
    a = np.zeros(n_samples)
    mask_multi = n_in > 1
    a[mask_multi] = (a_with_self[mask_multi] * n_in[mask_multi]) / (n_in[mask_multi] - 1)

    b_matrix = cluster_distances.copy()
    b_matrix[np.arange(n_samples), label_indices] = np.inf
    b = np.min(b_matrix, axis=1)

    denom = np.maximum(a, b)
    s = np.zeros(n_samples)
    
    mask_valid = (denom > 0) & (n_in > 1)
    s[mask_valid] = (b[mask_valid] - a[mask_valid]) / denom[mask_valid]

    return np.mean(s)