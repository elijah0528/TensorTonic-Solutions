import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    X, y = np.array(X), np.array(y)
    N = len(X)
    indices = np.arange(N)
    rng = np.random.default_rng(rng)
    rng.shuffle(indices)
    for start in range(0, N, batch_size):
        if drop_last and start + batch_size > N:
            break
        curr_indices = indices[start: start + batch_size]
        yield X[curr_indices], y[curr_indices]