import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    dk = K.shape[-1]
    scores = Q @ K.transpose(-2, -1)
    norm_scores = scores / math.sqrt(dk)
    output = F.softmax(norm_scores, dim = -1)
    return output @ V