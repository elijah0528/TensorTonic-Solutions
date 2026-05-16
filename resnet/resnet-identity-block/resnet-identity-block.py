import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    x, W1, W2 = np.array(x), np.array(W1), np.array(W2)
    output = np.maximum(x @ W1.T, 0)
    output = np.maximum(output @ W2.T + x, 0)
    return output
    
