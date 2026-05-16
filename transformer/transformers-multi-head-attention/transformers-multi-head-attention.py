import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    
    B, T, d_model = Q.shape
    d_head = d_model // num_heads
    # Multiply by weights
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v
    # Reshape
    Q = Q.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3) 
    V = V.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) # (B, T, dk, dk)
    normed = scores / np.sqrt(d_head)
    s_normed = softmax(normed)
    output = s_normed @ V
    output = output.transpose(0, 2, 1, 3).reshape(B, T, d_model)
    output = output @ W_o
    return output
    

    