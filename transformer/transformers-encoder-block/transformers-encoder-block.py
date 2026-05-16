import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    num = x - np.mean(x, axis = -1, keepdims = True)
    denom = np.sqrt(np.var(x, axis = -1, keepdims = True) + eps)
    return gamma * (num / denom) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    B, T, n_dims = Q.shape
    head_dim = n_dims // num_heads

    # B, num_heads, T, head_dim
    Q = Q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) # (B, T, head_dim, head_dim)
    scores = scores / np.sqrt(head_dim)
    attn = softmax(scores)
    output = attn @ V  # (B, T, head_dim, head_dim)
    output = output.transpose(0, 2, 1, 3).reshape(B, T, n_dims)
    output = output @ W_o # (B, num_heads, T, head_dim)
    return output
    


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    l1 = x @ W1 + b1
    norm = np.maximum(l1, 0)
    l2 = norm @ W2 + b2
    return l2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attn = multi_head_attention(
        x, x, x, W_q, W_k, W_v, W_o, num_heads
    )
    x = layer_norm(x + attn, gamma1, beta1)

    ffn = feed_forward(
        x, W1, b1, W2, b2
    )
    x = layer_norm(x + ffn, gamma2, beta2)

    return x
    
