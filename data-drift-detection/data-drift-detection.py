import numpy as np

def detect_drift(reference_counts, production_counts, threshold):
    ref = np.array(reference_counts, dtype=float)
    prod = np.array(production_counts, dtype=float)
    
    p = ref / np.sum(ref)
    q = prod / np.sum(prod)
    
    score = 0.5 * np.sum(np.abs(p - q))
    drift_detected = score > threshold
    
    return {
        "score": score,
        "drift_detected": bool(drift_detected)
    }