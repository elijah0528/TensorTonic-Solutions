import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    rng = np.random.default_rng(rng)
    mask = (rng.random(x.shape) > p)
    dropout_pattern = mask.astype(float) / (1 - p)

    output = x * dropout_pattern

    return output, dropout_pattern