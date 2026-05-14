import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = 0
        for seq in seqs:
            max_len = max(max_len, len(seq))
    res = []
    for seq in seqs:
        # Truncate
        row = np.array(seq)
        
        if len(seq) > max_len:
            row = row[:max_len]
        # If truncated its the right shape
        # Diff is 0 unless it needs padding
        diff = max_len - len(row)
        if diff > 0:
            padding = np.full((diff, ), pad_value)
            row = np.concatenate([row, padding])
        res.append(row)
    return np.array(res)
            
            
            