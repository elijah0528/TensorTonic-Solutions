import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    T = scores.shape[-1]
    mask = np.tril(np.ones((T, T))) # (T, T)
    masked_scores = np.where(mask == 1, scores, mask_value)
    return masked_scores