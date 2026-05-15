import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    seq = np.arange(seq_len)[:, None]
    dims = np.arange(d_model)[None, :]
    # Denom is a numpy array
    denom = base ** ((2 * (dims// 2)) / d_model)
    angles = seq / denom
    pe_1 = np.sin(angles)
    pe_2 = np.cos(angles)

    mask_1 = (dims % 2 == 0)
    mask_2 = (dims % 2 == 1)

    res = pe_1 * mask_1 + pe_2 * mask_2
    return res
    