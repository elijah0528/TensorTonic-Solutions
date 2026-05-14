import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    # m examples with n features
    m, n = X.shape # X -> (m, n)
    w = np.zeros(n) # (n, )
    b = 0.0
    for i in range(steps):
        # X @ w -> (m, )
        z = X @ w + b
        # Activation function
        y_pred = _sigmoid(z) # (m, )
        # Total Loss (J) is neg log likelihood
        
        # dJ / dz = y_pred - y
        error = y_pred - y # (m, )
        # dJ / dw = dJ / dz * dz / dw
        dw = (X.T @ error)
        # dJ / db = dJ / dz * dz / db
        db = np.mean(error)

        # Gradient update
        w -= dw * lr
        b -= db * lr
    
    return (w, b)

        
        