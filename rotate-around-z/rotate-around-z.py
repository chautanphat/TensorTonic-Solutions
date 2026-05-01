import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    points = np.array(points)
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    W = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
    points = W @ points.T
    
    return points.T