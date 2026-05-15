import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    if axis == 0:
        return np.array(arr, dtype = np.float64)[indices, :]
    else:
        return np.array(arr, dtype = np.float64)[:, indices]

    