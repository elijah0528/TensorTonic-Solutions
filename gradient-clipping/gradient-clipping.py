import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.array(g)
    g_squared = g ** 2
    s = g_squared.sum()
    norm = np.sqrt(s)
    if max_norm <= 0: 
        return g

    return g if norm < max_norm else g * (max_norm / norm)