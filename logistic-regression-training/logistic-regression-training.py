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
    w = np.zeros(X.shape[1])
    b = 0.0
    for i in range(steps):
        p = _sigmoid(X @ w + b)
        wgradient = (1/len(y)) *  X.T @ (p-y) 
        bgradient = np.mean(p-y)
        w -= lr * wgradient
        b -= lr * bgradient
    return (w,b)