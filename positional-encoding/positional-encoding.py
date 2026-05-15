import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    seq = np.arange(seq_len)[:, None]
    # Denom is a numpy array
    denom = base ** ((2 * (np.arange(d_model)// 2)) / d_model)
    angles = seq / denom

    pe = np.empty((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe
    