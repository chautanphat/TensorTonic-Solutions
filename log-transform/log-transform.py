import math

def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    return [math.log(1 + value) for value in values]