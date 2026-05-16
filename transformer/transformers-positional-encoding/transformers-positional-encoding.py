import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    seq = np.arange(seq_length)[:, None]
    dims = np.arange(0, d_model, 2)[None, :]
    denom = 10_000 ** (dims / d_model)
    pe = np.zeros((seq_length, d_model))
    angles = np.array(seq / denom, dtype = np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe