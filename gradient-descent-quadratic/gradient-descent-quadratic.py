def _gradient(a, b, x):
    return 2 * a * x + b

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    for t in range(steps):
        grad = _gradient(a, b, x0)
        x0 = x0 - lr * grad
    return x0